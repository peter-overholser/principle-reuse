"""
Small transformer (and an LSTM comparison family) for the model panel.

Hand-rolled rather than nn.TransformerEncoder so that the residual stream is
directly addressable: the causal tier needs to read and write activations at a
chosen layer and position.

Devices: prefers CUDA, then Apple MPS, then CPU.
"""

import os

import torch
import torch.nn as nn

# nn.MultiheadAttention takes a fused fast path when need_weights=False, and that
# path has had correctness problems on some backends (notably MPS) when a
# key_padding_mask is supplied. SIMPLE_ATTN swaps in a longhand implementation
# that avoids it entirely.  PANEL_SIMPLE_ATTN=1 to force it on.
SIMPLE_ATTN = os.environ.get("PANEL_SIMPLE_ATTN", "0") == "1"


def get_device(pref=None):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleAttention(nn.Module):
    """Multi-head self-attention written out longhand. No fused kernels."""

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0, f"d={d} not divisible by heads={heads}"
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2)
                   for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, heads, p=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.simple = SIMPLE_ATTN
        self.attn = (SimpleAttention(d, heads) if self.simple
                     else nn.MultiheadAttention(d, heads, dropout=p,
                                                batch_first=True))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        if self.simple:
            a = self.attn(h, key_padding_mask=key_padding_mask)
        else:
            a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                             need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class ResidualBottleneck(nn.Module):
    """A fixed-rank, trainable channel shared across residual applications.

    Every V3 condition instantiates the same d-by-d down/up parameterization.
    ``active_dim`` changes only the number of latent coordinates that can carry
    information, keeping the nominal parameter count constant across pressure
    conditions.
    """

    def __init__(self, d, active_dim):
        super().__init__()
        if not 1 <= active_dim <= d:
            raise ValueError(f"bottleneck_dim must be in [1, {d}], got {active_dim}")
        self.d, self.active_dim = d, active_dim
        self.down = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, d, bias=False)
        self.register_buffer(
            "active_mask",
            torch.cat([torch.ones(active_dim), torch.zeros(d - active_dim)]),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            identity = torch.eye(
                self.d, device=self.down.weight.device, dtype=self.down.weight.dtype
            )
            self.down.weight.copy_(identity)
            self.up.weight.copy_(identity)

    def reset_random_projection(self, seed):
        """Replace the learned channel with a seeded, well-conditioned subspace.

        For a tight channel this destroys the learned basis without changing its
        rank or singular values.  At full rank the down/up product is identity,
        which is a useful sanity property for tests.
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        random_matrix = torch.randn(self.d, self.d, generator=generator)
        orthogonal, _ = torch.linalg.qr(random_matrix)
        orthogonal = orthogonal.to(
            device=self.down.weight.device, dtype=self.down.weight.dtype
        )
        with torch.no_grad():
            self.down.weight.copy_(orthogonal.T)
            self.up.weight.copy_(orthogonal)

    def encode(self, x):
        return self.down(x) * self.active_mask.to(dtype=x.dtype)

    def forward(self, x):
        return self.up(self.encode(x))


class ResidualAdapter(nn.Module):
    """Zero-output residual adapter used by the staged V6 experiment.

    The down projection is initialized normally while the up projection is
    exactly zero. Enabling a dormant adapter therefore leaves the network
    function unchanged at the consolidation boundary, but gives the first
    optimizer update a non-zero gradient for the up projection.
    """

    def __init__(self, d, rank):
        super().__init__()
        if not 1 <= rank <= d:
            raise ValueError(f"adapter rank must be in [1, {d}], got {rank}")
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank)
        self.up = nn.Linear(rank, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(torch.nn.functional.gelu(self.down(self.norm(x))))


class Transformer(nn.Module):
    family = "transformer"

    def __init__(
        self, vocab, seq_len, pad, d=96, layers=3, heads=4, p=0.0,
        bottleneck_dim=0, adapter_rank=0,
    ):
        super().__init__()
        self.pad, self.d, self.layers_n = pad, d, layers
        self.n_resid = layers          # one addressable residual per block
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, p) for _ in range(layers)])
        self.bottleneck = (
            ResidualBottleneck(d, bottleneck_dim) if bottleneck_dim else None
        )
        self.adapter_rank = int(adapter_rank)
        self.adapters = nn.ModuleList(
            [ResidualAdapter(d, self.adapter_rank) for _ in range(layers)]
            if self.adapter_rank else []
        )
        self.adapters_enabled = False
        self.delta_readout_enabled = False
        self.residual_adapter_gate_token = None
        self.delta_readout_gate_token = None
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.delta_head = (
            nn.Linear(d, 1) if self.adapter_rank else None
        )
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    def apply_bottleneck(self, x):
        return x if self.bottleneck is None else self.bottleneck(x)

    def effective_embedding_weights(self):
        return self.apply_bottleneck(self.emb.weight)

    def reset_bottleneck(self, seed):
        if self.bottleneck is None:
            raise RuntimeError("cannot reset a model without a residual bottleneck")
        self.bottleneck.reset_random_projection(seed)

    def configure_adapters(
        self, residual: bool, readout: bool, *,
        residual_gate_token=None, readout_gate_token=None,
    ) -> None:
        """Configure adapter paths and optional observable-context gates.

        A gated path is active only for sequences containing its gate token.
        The optional gates are used by V7; omitting them preserves V6 behavior.
        """
        if (residual or readout) and not self.adapter_rank:
            raise RuntimeError("cannot enable adapters when adapter_rank=0")
        self.adapters_enabled = bool(residual)
        self.delta_readout_enabled = bool(readout)
        self.residual_adapter_gate_token = (
            None if residual_gate_token is None else int(residual_gate_token)
        )
        self.delta_readout_gate_token = (
            None if readout_gate_token is None else int(readout_gate_token)
        )

    @staticmethod
    def _sequence_gate(idx, token, dtype):
        return idx.eq(int(token)).any(dim=1).to(dtype=dtype)

    def adapter_parameters(self):
        """Iterate parameters belonging to V6's added trainable capacity."""
        yield from self.adapters.parameters()
        if self.delta_head is not None:
            yield from self.delta_head.parameters()

    @torch.no_grad()
    def adapter_diagnostics(self):
        squared = 0.0
        count = 0
        for parameter in self.adapter_parameters():
            values = parameter.detach().float()
            squared += float(torch.sum(values * values).cpu())
            count += parameter.numel()
        return {
            "adapter_rank": float(self.adapter_rank),
            "adapter_parameter_count": float(count),
            "adapter_parameter_l2": squared ** 0.5,
            "residual_adapters_enabled": float(self.adapters_enabled),
            "delta_readout_enabled": float(self.delta_readout_enabled),
            "residual_adapter_context_gated": float(
                self.residual_adapter_gate_token is not None
            ),
            "delta_readout_context_gated": float(
                self.delta_readout_gate_token is not None
            ),
        }

    def residuals(self, idx):
        """Forward pass returning the residual stream after every block."""
        mask = idx.eq(self.pad)
        x = self.apply_bottleneck(self.emb(idx) + self.pos[:, :idx.size(1)])
        outs = []
        for index, b in enumerate(self.blocks):
            x = self.apply_bottleneck(b(x, key_padding_mask=mask))
            if self.adapters_enabled:
                update = self.adapters[index](x)
                if self.residual_adapter_gate_token is not None:
                    gate = self._sequence_gate(
                        idx, self.residual_adapter_gate_token, update.dtype
                    )[:, None, None]
                    update = update * gate
                x = x + update
            outs.append(x)
        return outs, mask

    def readout(self, x, mask, idx=None):
        """
        Read out at the LAST non-pad position, which is where the query ends.

        An earlier version applied ln_f per position and then mean-pooled. That
        silently caps learning: LayerNorm forces every position to unit scale,
        so the model cannot amplify the three query tokens against the ~30
        context tokens, and the label here lives entirely in the query's
        argument order. Every model in the first panel sat at loss ln(2).
        """
        last = (~mask).sum(1) - 1
        h = x[torch.arange(x.size(0), device=x.device), last]
        normalized = self.ln_f(h)
        output = self.head(normalized).squeeze(-1)
        if self.delta_readout_enabled:
            delta = self.delta_head(normalized).squeeze(-1)
            if self.delta_readout_gate_token is not None:
                if idx is None:
                    raise RuntimeError("a gated delta readout requires input tokens")
                delta = delta * self._sequence_gate(
                    idx, self.delta_readout_gate_token, delta.dtype
                )
            output = output + delta
        return output

    def forward(self, idx):
        outs, mask = self.residuals(idx)
        return self.readout(outs[-1], mask, idx)

    def forward_from(self, idx, layer, resid):
        """Continue the forward pass from a modified residual stream at `layer`."""
        mask = idx.eq(self.pad)
        # ``resid`` is already the output of this layer's bottleneck. Applying
        # it again would make causal interventions follow a different forward
        # path from the ordinary model.
        x = resid
        for index, b in enumerate(self.blocks[layer + 1:], start=layer + 1):
            x = self.apply_bottleneck(b(x, key_padding_mask=mask))
            if self.adapters_enabled:
                x = x + self.adapters[index](x)
        return self.readout(x, mask, idx)


class LSTMModel(nn.Module):
    """Second architecture family, so the analysis can hold a family out."""
    family = "lstm"

    def __init__(
        self, vocab, seq_len, pad, d=96, layers=2, heads=None, p=0.0,
        bottleneck_dim=0, adapter_rank=0,
    ):
        super().__init__()
        if adapter_rank:
            raise ValueError("residual adapters are currently transformer-only")
        self.pad, self.d, self.layers_n = pad, d, layers
        self.n_resid = 1               # residuals() exposes one hidden state
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=p if layers > 1 else 0.0)
        self.proj = nn.Linear(2 * d, d)
        self.bottleneck = (
            ResidualBottleneck(d, bottleneck_dim) if bottleneck_dim else None
        )
        self.head = nn.Linear(d, 1)

    def apply_bottleneck(self, x):
        return x if self.bottleneck is None else self.bottleneck(x)

    def effective_embedding_weights(self):
        return self.apply_bottleneck(self.emb.weight)

    def reset_bottleneck(self, seed):
        if self.bottleneck is None:
            raise RuntimeError("cannot reset a model without a residual bottleneck")
        self.bottleneck.reset_random_projection(seed)

    def residuals(self, idx):
        mask = idx.eq(self.pad)
        h, _ = self.rnn(self.apply_bottleneck(self.emb(idx)))
        return [self.apply_bottleneck(self.proj(h))], mask

    def readout(self, x, mask, idx=None):
        """Last non-pad position, matching the transformer."""
        last = (~mask).sum(1) - 1
        h = x[torch.arange(x.size(0), device=x.device), last]
        return self.head(h).squeeze(-1)

    def forward(self, idx):
        outs, mask = self.residuals(idx)
        return self.readout(outs[-1], mask, idx)

    def forward_from(self, idx, layer, resid):
        mask = idx.eq(self.pad)
        return self.readout(resid, mask, idx)


def build(family, vocab, seq_len, pad, **kw):
    cls = {"transformer": Transformer, "lstm": LSTMModel}[family]
    return cls(vocab, seq_len, pad, **kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters())

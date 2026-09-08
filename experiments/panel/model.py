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


class Transformer(nn.Module):
    family = "transformer"

    def __init__(self, vocab, seq_len, pad, d=96, layers=3, heads=4, p=0.0):
        super().__init__()
        self.pad, self.d, self.layers_n = pad, d, layers
        self.n_resid = layers          # one addressable residual per block
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, p) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def residuals(self, idx):
        """Forward pass returning the residual stream after every block."""
        mask = idx.eq(self.pad)
        x = self.emb(idx) + self.pos[:, :idx.size(1)]
        outs = []
        for b in self.blocks:
            x = b(x, key_padding_mask=mask)
            outs.append(x)
        return outs, mask

    def readout(self, x, mask):
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
        return self.head(self.ln_f(h)).squeeze(-1)

    def forward(self, idx):
        outs, mask = self.residuals(idx)
        return self.readout(outs[-1], mask)

    def forward_from(self, idx, layer, resid):
        """Continue the forward pass from a modified residual stream at `layer`."""
        mask = idx.eq(self.pad)
        x = resid
        for b in self.blocks[layer + 1:]:
            x = b(x, key_padding_mask=mask)
        return self.readout(x, mask)


class LSTMModel(nn.Module):
    """Second architecture family, so the analysis can hold a family out."""
    family = "lstm"

    def __init__(self, vocab, seq_len, pad, d=96, layers=2, heads=None, p=0.0):
        super().__init__()
        self.pad, self.d, self.layers_n = pad, d, layers
        self.n_resid = 1               # residuals() exposes one hidden state
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=p if layers > 1 else 0.0)
        self.proj = nn.Linear(2 * d, d)
        self.head = nn.Linear(d, 1)

    def residuals(self, idx):
        mask = idx.eq(self.pad)
        h, _ = self.rnn(self.emb(idx))
        return [self.proj(h)], mask

    def readout(self, x, mask):
        """Last non-pad position, matching the transformer."""
        last = (~mask).sum(1) - 1
        h = x[torch.arange(x.size(0), device=x.device), last]
        return self.head(h).squeeze(-1)

    def forward(self, idx):
        outs, mask = self.residuals(idx)
        return self.readout(outs[-1], mask)

    def forward_from(self, idx, layer, resid):
        mask = idx.eq(self.pad)
        return self.readout(resid, mask)


def build(family, vocab, seq_len, pad, **kw):
    cls = {"transformer": Transformer, "lstm": LSTMModel}[family]
    return cls(vocab, seq_len, pad, **kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters())

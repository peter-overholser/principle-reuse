"""Evaluate saved checkpoints on the test-combination and held-pair splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.panel import model as model_library
from experiments.training_axes import metrics, task, task_differences
from experiments.training_axes import task_v8


ROOT = Path(__file__).resolve().parent
TASKS = {
    "order": task,
    "differences": task_differences,
    "order_v8": task_v8.ORDER,
    "differences_v8": task_v8.DIFFERENCES,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-size", type=int, default=2000)
    args = parser.parse_args()

    paths = sorted(args.checkpoint_dir.glob("step-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found under {args.checkpoint_dir}")
    output = args.out or (ROOT / "locked_results" / f"{args.checkpoint_dir.name}.jsonl")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = model_library.get_device(args.device)
    first_saved = torch.load(paths[0], map_location=device, weights_only=True)
    task_name = first_saved["config"].get("task_name", "order")
    task_module = TASKS[task_name]
    dataset_kwargs = {
        "seed": 0, "n": args.eval_size,
        "train_dists": first_saved["config"]["train_dists"],
    }
    if first_saved["config"].get("v8_arm"):
        dataset_kwargs["include_locked"] = True
    datasets = task_module.evaluation_sets(**dataset_kwargs)

    with output.open("w") as handle:
        for path in paths:
            saved = torch.load(path, map_location=device, weights_only=True)
            config = saved["config"]
            model = model_library.build(
                config["family"], task_module.VOCAB, task_module.SEQ_LEN,
                task_module.PAD,
                d=config["width"], layers=config["layers"],
                bottleneck_dim=config.get("bottleneck_dim", 0),
            ).to(device)
            model.load_state_dict(saved["model"])
            result = metrics.evaluate_checkpoint(
                model, datasets, device, seed=config["seed"] + saved["step"],
                include_locked=True,
                task_module=task_module,
            )
            locked = {
                key: value for key, value in result.items()
                if key.startswith(("test_combo_", "held_pair_", "test_both_"))
            }
            row = {
                "config_id": config["config_id"], "step": saved["step"], **locked
            }
            handle.write(json.dumps(row, allow_nan=True) + "\n")
            print(
                f"step={saved['step']:>6} "
                f"test={locked['test_combo_accuracy']:.3f} "
                f"both={locked['test_both_accuracy']:.3f}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

"""Run the Adap-CTA adaptive reconstruction attack on one captured bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flcore.attacks.adaptive import (
    AdaptiveAttack,
    AdaptiveAttackConfig,
    load_adaptive_observation,
)
from flcore.attacks.gi_nas.bundle import build_victim_model


def _metrics(reference: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    reconstruction = reconstruction.float().clamp(float(reference.min()), float(reference.max()))
    mse = float(torch.mean((reference - reconstruction) ** 2))
    peak = max(float(reference.max() - reference.min()), 1e-6)
    return {"mse": mse, "psnr": float(10.0 * torch.log10(torch.tensor(peak * peak / max(mse, 1e-12))))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="resnet", help="Model name used by main.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    args = parser.parse_args()

    bundle = load_adaptive_observation(args.bundle)
    attack = bundle["attack"]
    num_classes = int(attack.get("num_classes", 10))
    model = build_victim_model(args.model, num_classes).to(args.device)
    model.load_state_dict(attack["model_state_dict"])
    images, stats = AdaptiveAttack(
        AdaptiveAttackConfig(
            iterations=args.iterations,
            learning_rate=args.lr,
            restarts=args.restarts,
            tv_weight=args.tv_weight,
            seed=args.seed,
        )
    ).reconstruct(model, bundle)

    output_dir = args.output_dir or args.bundle.parent / (args.bundle.stem + "_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(images.cpu(), output_dir / "reconstructed.pt")
    evaluation = bundle.get("evaluation_only", {})
    if "images" in evaluation:
        stats.update({f"image_{key}": value for key, value in _metrics(evaluation["images"], images.cpu()).items()})
    try:
        from torchvision.utils import save_image

        low, high = float(images.min()), float(images.max())
        display = ((images - low) / max(high - low, 1e-6)).clamp(0, 1)
        save_image(display, output_dir / "reconstructed_grid.png", nrow=max(1, min(8, images.shape[0])))
    except Exception as error:
        stats["grid_export_error"] = str(error)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

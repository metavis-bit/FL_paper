"""Command-line runner for the project-integrated GI-NAS attack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision.utils as vutils

from flcore.attacks.gi_nas.attack import GINASAttack, GINASConfig
from flcore.attacks.gi_nas.bundle import (
    build_victim_model,
    evaluate_reconstruction,
    load_gradient_bundle,
    require_gradient_protocol,
)
from flcore.attacks.gi_nas.official.utils.reconstructed import convert_relu_to_sigmoid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(value)


def _json_safe(value):
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run original GI-NAS on an FL gradient bundle")
    parser.add_argument("--bundle", required=True, help="Path to a bundle exported by export_attack_bundle.py")
    parser.add_argument(
        "--output-dir", default="privacy/gi_nas_results", help="Directory for images and metrics"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--search-size", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=0)
    parser.add_argument("--no-lr-decay", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--allow-approximate-observation", action="store_true")
    parser.add_argument("--require-perceptual-metrics", action="store_true")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    bundle = load_gradient_bundle(bundle_path)
    protocol = require_gradient_protocol(
        bundle, allow_approximate=args.allow_approximate_observation
    )
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_victim_model(bundle["model_name"], int(bundle["num_classes"])).to(device)
    if bool(bundle.get("metadata", {}).get("relu_to_sigmoid", False)):
        convert_relu_to_sigmoid(model)
    model.load_state_dict(bundle["model_state_dict"], strict=True)

    attack = GINASAttack(
        GINASConfig(
            search_size=args.search_size,
            iterations=args.iterations,
            learning_rate=args.lr,
            tv_weight=args.tv_weight,
            seed=args.seed,
            report_every=args.report_every,
            learning_rate_decay=not args.no_lr_decay,
            verbose=not args.quiet,
        )
    )
    reconstructed, stats = attack.reconstruct(
        model,
        bundle["target_gradient"],
        bundle["labels"],
        bundle["image_shape"],
        bundle["normalization_mean"],
        bundle["normalization_std"],
    )
    stats["attack"] = "gi_nas"
    stats["implementation"] = "project-integrated upstream GI-NAS"
    stats["threat_model"] = protocol
    stats["label_mode"] = "known_ground_truth_multiset"
    stats["seed"] = args.seed
    stats["config"] = {
        "search_size": args.search_size,
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "tv_weight": args.tv_weight,
        "learning_rate_decay": not args.no_lr_decay,
    }
    stats["provenance"] = {
        "bundle": str(bundle_path.resolve()),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_metadata": bundle.get("metadata", {}),
        "model_name": bundle["model_name"],
        "num_classes": int(bundle["num_classes"]),
        "batch_size": int(bundle["labels"].shape[0]),
        "labels": bundle["labels"].tolist(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vutils.save_image(reconstructed, output_dir / "reconstructed_grid.png", nrow=reconstructed.shape[0])

    metrics = None
    if "images_01" in bundle:
        _, metrics = evaluate_reconstruction(
            bundle["images_01"],
            reconstructed,
            bundle["labels"],
            model,
            bundle["normalization_mean"],
            bundle["normalization_std"],
        )
        vutils.save_image(
            bundle["images_01"].clamp(0, 1),
            output_dir / "original_grid.png",
            nrow=bundle["images_01"].shape[0],
        )
        stats["metrics"] = metrics
        if args.require_perceptual_metrics and (
            metrics.get("lpips_mean") is None or metrics.get("fsim_mean") is None
        ):
            raise RuntimeError(
                "LPIPS/FSIM were required but unavailable: "
                f"{metrics.get('perceptual_metric_error')}"
            )

    stats["resources"] = {
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "restarts": 1,
    }

    with (output_dir / "reconstructed.pt").open("wb") as handle:
        torch.save(reconstructed, handle)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(stats), handle, indent=2, ensure_ascii=False)

    print(f"GI-NAS reconstruction saved to: {output_dir.resolve()}")
    if metrics:
        print(
            "MSE={:.6f}, PSNR={:.3f} dB, SSIM={}, recognition={:.3f}".format(
                metrics["mse_mean"],
                metrics["psnr_db_mean"] if metrics["psnr_db_mean"] is not None else float("nan"),
                "N/A" if metrics["ssim_mean"] is None else f"{metrics['ssim_mean']:.4f}",
                metrics["recognition_rate"],
            )
        )


if __name__ == "__main__":
    main()

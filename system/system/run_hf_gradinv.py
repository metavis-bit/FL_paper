"""Run HF-GradInv against a gradient bundle exported by ``export_attack_bundle.py``.

The runner intentionally hides the ground-truth labels from the attack.  The
labels stored in a bundle are used only for post-attack metrics unless
``--use-ground-truth-labels`` is explicitly selected for a controlled upper
bound/debug run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.utils as vutils

from flcore.attacks.gi_nas.bundle import (
    build_victim_model,
    evaluate_reconstruction,
    label_multiset_score,
    load_gradient_bundle,
    require_gradient_protocol,
)
from flcore.attacks.hf_gradinv import HFGradInvAttack, HFGradInvConfig


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


def _load_auxiliary(path: str | Path) -> torch.Tensor:
    """Load public auxiliary images from ``.pt``, ``.npy`` or ``.npz``."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("images_01", "images", "data", "x"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, torch.Tensor):
            raise ValueError("auxiliary .pt file must contain a tensor or a tensor-valued mapping")
        images = value.detach().cpu().float()
    elif suffix == ".npy":
        images = torch.from_numpy(np.load(path)).float()
    elif suffix == ".npz":
        archive = np.load(path)
        if not archive.files:
            raise ValueError("auxiliary .npz file is empty")
        key = next((candidate for candidate in ("images_01", "images", "data", "x") if candidate in archive), archive.files[0])
        images = torch.from_numpy(archive[key]).float()
    else:
        raise ValueError("auxiliary data must be a .pt/.pth, .npy or .npz file")

    if images.ndim != 4:
        raise ValueError(f"auxiliary images must have shape [N,C,H,W], got {tuple(images.shape)}")
    if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
        images = images.permute(0, 3, 1, 2).contiguous()
    if images.max() > 1.0 or images.min() < 0.0:
        # Public image arrays are commonly uint8 or [-1, 1].
        if images.min() >= -1.0 and images.max() <= 1.0:
            images = (images + 1.0) / 2.0
        else:
            images = images / 255.0
    return images.clamp(0, 1)


def _json_safe(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF-GradInv on an FL gradient bundle")
    parser.add_argument("--bundle", required=True, help="Path to a bundle exported by export_attack_bundle.py")
    parser.add_argument("--output-dir", default="privacy/hf_gradinv_results")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--gradient-dropout", type=float, default=0.1)
    parser.add_argument("--signed-gradients", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=0)
    parser.add_argument(
        "--auxiliary-data",
        help="Optional disjoint public images (.pt/.npy/.npz) for CV-assisted label inference",
    )
    parser.add_argument(
        "--use-ground-truth-labels",
        action="store_true",
        help="Known-label protocol used for a fair comparison with GI-NAS",
    )
    parser.add_argument("--allow-approximate-observation", action="store_true")
    parser.add_argument("--require-perceptual-metrics", action="store_true")
    args = parser.parse_args()

    bundle = load_gradient_bundle(args.bundle)
    protocol = require_gradient_protocol(
        bundle, allow_approximate=args.allow_approximate_observation
    )
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_victim_model(bundle["model_name"], int(bundle["num_classes"])).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)

    auxiliary_data = _load_auxiliary(args.auxiliary_data) if args.auxiliary_data else None
    attack = HFGradInvAttack(
        HFGradInvConfig(
            iterations=args.iterations,
            stages=args.stages,
            learning_rate=args.lr,
            restarts=args.restarts,
            tv_weight=args.tv_weight,
            gradient_dropout=args.gradient_dropout,
            signed_gradients=args.signed_gradients,
            seed=args.seed,
            report_every=args.report_every,
        )
    )
    reconstructed, stats = attack.reconstruct_from_bundle(
        model,
        bundle,
        auxiliary_data=auxiliary_data,
        use_ground_truth_labels=args.use_ground_truth_labels,
    )
    bundle_path = Path(args.bundle).resolve()
    stats["attack"] = "hf_gradinv_style_adapter"
    stats["implementation"] = "HF-GradInv-style project adapter (not author implementation)"
    stats["threat_model"] = protocol
    stats["label_mode"] = (
        "known_ground_truth_multiset" if args.use_ground_truth_labels else "inferred"
    )
    stats["seed"] = args.seed
    stats["config"] = {
        "iterations": args.iterations,
        "stages": args.stages,
        "learning_rate": args.lr,
        "restarts": args.restarts,
        "tv_weight": args.tv_weight,
        "gradient_dropout": args.gradient_dropout,
        "signed_gradients": args.signed_gradients,
        "auxiliary_data": str(Path(args.auxiliary_data).resolve()) if args.auxiliary_data else None,
    }
    stats["provenance"] = {
        "bundle": str(bundle_path),
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

    if "images_01" in bundle:
        _, metrics = evaluate_reconstruction(
            bundle["images_01"],
            reconstructed,
            bundle["labels"],
            model,
            bundle["normalization_mean"],
            bundle["normalization_std"],
        )
        stats["metrics"] = metrics
        if not args.use_ground_truth_labels:
            inferred_labels = torch.as_tensor(
                stats["label_inference"]["inferred_labels"], dtype=torch.long
            )
            metrics.update(label_multiset_score(bundle["labels"], inferred_labels))
        if args.require_perceptual_metrics and (
            metrics.get("lpips_mean") is None or metrics.get("fsim_mean") is None
        ):
            raise RuntimeError(
                "LPIPS/FSIM were required but unavailable: "
                f"{metrics.get('perceptual_metric_error')}"
            )
        vutils.save_image(
            bundle["images_01"].clamp(0, 1),
            output_dir / "original_grid.png",
            nrow=bundle["images_01"].shape[0],
        )

    stats["resources"] = {
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "restarts": int(args.restarts),
    }

    with (output_dir / "reconstructed.pt").open("wb") as handle:
        torch.save(reconstructed, handle)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(stats), handle, indent=2, ensure_ascii=False)

    print(f"HF-GradInv reconstruction saved to: {output_dir.resolve()}")
    print(f"Inferred labels: {stats['label_inference']['inferred_labels']}")
    if "metrics" in stats:
        metrics = stats["metrics"]
        print(
            "MSE={:.6f}, PSNR={}, SSIM={}, recognition={:.3f}".format(
                metrics["mse_mean"],
                "N/A" if metrics["psnr_db_mean"] is None else f"{metrics['psnr_db_mean']:.3f} dB",
                "N/A" if metrics["ssim_mean"] is None else f"{metrics['ssim_mean']:.4f}",
                metrics["recognition_rate"],
            )
        )
if __name__ == "__main__":
    main()

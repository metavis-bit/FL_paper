"""Serialization and evaluation helpers for GI-NAS experiments."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BUNDLE_FORMAT_VERSION = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_multiset_score(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
    """Score an inferred label multiset without assuming reconstruction order."""

    expected = [int(value) for value in reference.detach().cpu().flatten().tolist()]
    inferred = [int(value) for value in candidate.detach().cpu().flatten().tolist()]
    if not expected:
        raise ValueError("reference label multiset must not be empty")
    expected_counts = Counter(expected)
    inferred_counts = Counter(inferred)
    overlap = sum(
        min(count, inferred_counts.get(label, 0))
        for label, count in expected_counts.items()
    )
    return {
        "label_multiset_accuracy": float(overlap / len(expected)),
        "label_multiset_exact_match": expected_counts == inferred_counts,
        "inferred_label_count": len(inferred),
        "expected_label_count": len(expected),
    }


def preflight_perceptual_metrics() -> Dict[str, Any]:
    """Exercise the exact FSIM/LPIPS backend before expensive attack jobs."""

    import piq

    sample = torch.linspace(0.0, 1.0, 3 * 32 * 32, dtype=torch.float32).reshape(
        1, 3, 32, 32
    )
    with torch.no_grad():
        fsim = piq.fsim(sample, sample, data_range=1.0)
        lpips = piq.LPIPS(reduction="none")(sample, sample)
    if not torch.isfinite(fsim).all() or not torch.isfinite(lpips).all():
        raise RuntimeError("PIQ perceptual metric preflight returned a non-finite value")
    return {
        "backend": f"piq-{getattr(piq, '__version__', 'unknown')}",
        "fsim_smoke": float(fsim.detach().float().mean()),
        "lpips_smoke": float(lpips.detach().float().mean()),
    }


def gradient_observation_protocol(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Return and validate the threat-model declaration for a gradient runner."""

    metadata = bundle.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    threat_model_id = str(
        metadata.get("threat_model_id", "A_legacy_standardized_exact_gradient")
    )
    observation_mode = str(metadata.get("observation_mode", "legacy_exact_gradient"))
    is_approximate = threat_model_id.startswith("C_") or bool(
        metadata.get("approximation_statement")
    )
    if threat_model_id.startswith("B_"):
        raise ValueError(
            "server-visible model deltas are not gradient bundles; use the "
            "client-update matching runner"
        )
    return {
        "threat_model_id": threat_model_id,
        "observation_mode": observation_mode,
        "is_approximate_gradient": is_approximate,
        "approximation_statement": metadata.get("approximation_statement"),
    }


def require_gradient_protocol(
    bundle: Mapping[str, Any], *, allow_approximate: bool = False
) -> Dict[str, Any]:
    protocol = gradient_observation_protocol(bundle)
    if protocol["is_approximate_gradient"] and not allow_approximate:
        raise ValueError(
            "this is a delta-derived approximate gradient (threat model C); "
            "pass --allow-approximate-observation only for an explicitly labelled C experiment"
        )
    return protocol


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_gradient_bundle(
    path: str | Path,
    model: nn.Module,
    model_name: str,
    num_classes: int,
    gradients: Sequence[torch.Tensor],
    labels: torch.Tensor,
    normalized_images: torch.Tensor,
    normalization_mean: Sequence[float] = (0.5, 0.5, 0.5),
    normalization_std: Sequence[float] = (0.5, 0.5, 0.5),
    metadata: Optional[Mapping[str, Any]] = None,
    model_state_reference: Optional[str | Path] = None,
    model_state_reference_sha256: Optional[str] = None,
) -> Path:
    """Save a victim state and gradient-like observation used by an attack."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = tuple(float(value) for value in normalization_mean)
    std = tuple(float(value) for value in normalization_std)
    mean_tensor = normalized_images.new_tensor(mean).view(1, -1, 1, 1)
    std_tensor = normalized_images.new_tensor(std).view(1, -1, 1, 1)
    images_01 = (normalized_images.detach() * std_tensor + mean_tensor).clamp(0, 1)

    payload = {
        "format": "fl-paper-gradient-observation-bundle",
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_name": str(model_name),
        "num_classes": int(num_classes),
        "target_gradient": [gradient.detach().cpu() for gradient in gradients],
        "labels": labels.detach().to(dtype=torch.long, device="cpu"),
        # Ground truth is stored strictly for post-attack metrics/figures.  The
        # reconstruction algorithm itself does not consume either image field.
        "normalized_images": normalized_images.detach().cpu(),
        "images_01": images_01.cpu(),
        "image_shape": tuple(int(value) for value in normalized_images.shape[1:]),
        "normalization_mean": mean,
        "normalization_std": std,
        "metadata": dict(metadata or {}),
    }
    if model_state_reference is None:
        payload["model_state_dict"] = _cpu_state_dict(model)
    else:
        reference = Path(model_state_reference)
        source = reference if reference.is_absolute() else path.parent / reference
        if not source.is_file():
            raise FileNotFoundError(f"referenced model state does not exist: {source}")
        payload["model_state_reference"] = str(reference)
        payload["model_state_reference_sha256"] = (
            model_state_reference_sha256 or _sha256(source)
        )
    torch.save(payload, path)
    return path


def load_gradient_bundle(path: str | Path) -> Dict[str, Any]:
    """Load and validate a shared gradient-observation bundle."""

    path = Path(path)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        bundle = torch.load(path, map_location="cpu")

    if "model_state_dict" not in bundle and "model_state_reference" in bundle:
        reference = Path(bundle["model_state_reference"])
        source = reference if reference.is_absolute() else path.parent / reference
        expected = bundle.get("model_state_reference_sha256")
        if not expected or _sha256(source) != expected:
            raise ValueError(f"referenced model state checksum mismatch: {source}")
        try:
            state = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(source, map_location="cpu")
        if "model_state_dict" not in state:
            raise ValueError(f"referenced file has no model_state_dict: {source}")
        bundle["model_state_dict"] = state["model_state_dict"]

    required = {
        "model_name",
        "num_classes",
        "model_state_dict",
        "target_gradient",
        "labels",
        "image_shape",
        "normalization_mean",
        "normalization_std",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"invalid gradient-observation bundle; missing fields: {', '.join(missing)}")
    if bundle.get("format_version", 0) > BUNDLE_FORMAT_VERSION:
        raise ValueError(
            "this gradient-observation bundle was created by a newer integration version"
        )
    return bundle


def build_victim_model(model_name: str, num_classes: int) -> nn.Module:
    """Recreate an image classifier supported by this project's main.py."""

    import torchvision

    normalized_name = model_name.lower()
    if normalized_name in {"resnet", "resnet18"}:
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if normalized_name == "resnet34":
        return torchvision.models.resnet34(weights=None, num_classes=num_classes)
    if normalized_name == "googlenet":
        return torchvision.models.googlenet(
            weights=None, aux_logits=False, num_classes=num_classes
        )
    if normalized_name == "alexnet":
        from flcore.trainmodel.alexnet import alexnet

        return alexnet(pretrained=False, num_classes=num_classes)
    if normalized_name == "vgg11":
        from flcore.trainmodel.VGG11 import VGG11

        return VGG11(num_classes=num_classes)
    if normalized_name == "mobilenet_v2":
        from flcore.trainmodel.mobilenet_v2 import mobilenet_v2

        return mobilenet_v2(pretrained=False, num_classes=num_classes)
    if normalized_name == "cnn":
        from flcore.trainmodel.models import FedAvgCNN

        return FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)
    if normalized_name == "resnet10":
        from flcore.trainmodel.resnet import resnet10

        return resnet10(num_classes=num_classes)
    raise ValueError(
        f"GI-NAS model builder does not support '{model_name}'. "
        "Use resnet/resnet18, resnet34, resnet10, VGG11, alexnet, "
        "googlenet, mobilenet_v2, or cnn."
    )


def _optimal_mse_assignment(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Tuple[torch.Tensor, Sequence[int]]:
    batch_size = original.shape[0]
    pairwise = torch.empty(batch_size, batch_size, dtype=torch.float64)
    for original_index in range(batch_size):
        for reconstructed_index in range(batch_size):
            pairwise[original_index, reconstructed_index] = F.mse_loss(
                original[original_index], reconstructed[reconstructed_index]
            )

    if batch_size <= 8:
        best_assignment = min(
            itertools.permutations(range(batch_size)),
            key=lambda assignment: sum(
                float(pairwise[index, candidate])
                for index, candidate in enumerate(assignment)
            ),
        )
    else:
        available = set(range(batch_size))
        greedy = []
        for original_index in range(batch_size):
            selected = min(
                available,
                key=lambda candidate: float(pairwise[original_index, candidate]),
            )
            available.remove(selected)
            greedy.append(selected)
        best_assignment = tuple(greedy)

    indices = torch.tensor(best_assignment, dtype=torch.long)
    return reconstructed[indices], list(best_assignment)


def evaluate_reconstruction(
    original_images_01: torch.Tensor,
    reconstructed_images_01: torch.Tensor,
    labels: torch.Tensor,
    victim_model: nn.Module,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Match the unordered batch and compute paper-facing reconstruction metrics."""

    original = original_images_01.detach().cpu().clamp(0, 1)
    reconstructed = reconstructed_images_01.detach().cpu().clamp(0, 1)
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"metric image shape mismatch: {tuple(original.shape)} vs "
            f"{tuple(reconstructed.shape)}"
        )
    reconstructed, assignment = _optimal_mse_assignment(original, reconstructed)

    mse_values = F.mse_loss(reconstructed, original, reduction="none").flatten(1).mean(1)
    psnr_values = [
        float("inf") if float(mse) == 0 else 10.0 * math.log10(1.0 / float(mse))
        for mse in mse_values
    ]

    ssim_values = []
    try:
        from flcore.servers.inversefed.pytorch_ssim_master import pytorch_ssim

        for index in range(original.shape[0]):
            value = pytorch_ssim.ssim(
                original[index : index + 1],
                reconstructed[index : index + 1],
            )
            ssim_values.append(float(value.detach().cpu()))
    except (ImportError, RuntimeError, ValueError):
        ssim_values = []

    try:
        device = next(victim_model.parameters()).device
    except StopIteration as exc:
        raise ValueError("victim_model has no parameters") from exc
    mean_tensor = reconstructed.new_tensor(normalization_mean).view(1, -1, 1, 1)
    std_tensor = reconstructed.new_tensor(normalization_std).view(1, -1, 1, 1)
    victim_input = ((reconstructed - mean_tensor) / std_tensor).to(device)
    labels = labels.detach().to(device=device, dtype=torch.long)
    previous_training_mode = victim_model.training
    victim_model.eval()
    with torch.no_grad():
        probabilities = F.softmax(victim_model(victim_input), dim=1)
        confidence, predictions = probabilities.max(dim=1)
    victim_model.train(previous_training_mode)

    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    metrics: Dict[str, Any] = {
        "matching": "minimum batch MSE",
        "reconstruction_assignment": assignment,
        "mse_per_image": [float(value) for value in mse_values],
        "mse_mean": float(mse_values.mean()),
        "psnr_db_per_image": psnr_values,
        "psnr_db_mean": (
            float("inf") if not finite_psnr else sum(finite_psnr) / len(finite_psnr)
        ),
        "recognition_rate": float((predictions == labels).float().mean().cpu()),
        "classifier_confidence_mean": float(confidence.mean().cpu()),
    }
    if ssim_values:
        metrics["ssim_per_image"] = ssim_values
        metrics["ssim_mean"] = sum(ssim_values) / len(ssim_values)
    else:
        metrics["ssim_mean"] = None

    metrics["fsim_mean"] = None
    metrics["lpips_mean"] = None
    metrics["perceptual_metric_backend"] = None
    metrics["perceptual_metric_error"] = None
    try:
        import piq

        fsim_values = []
        lpips_values = []
        lpips_metric = piq.LPIPS(reduction="none")
        for index in range(original.shape[0]):
            original_image = original[index : index + 1]
            reconstructed_image = reconstructed[index : index + 1]
            fsim_values.append(
                float(
                    piq.fsim(
                        original_image,
                        reconstructed_image,
                        data_range=1.0,
                    ).detach().cpu()
                )
            )
            lpips_value = lpips_metric(original_image, reconstructed_image)
            lpips_values.append(float(lpips_value.detach().float().mean().cpu()))
        metrics["fsim_per_image"] = fsim_values
        metrics["fsim_mean"] = sum(fsim_values) / len(fsim_values)
        metrics["lpips_per_image"] = lpips_values
        metrics["lpips_mean"] = sum(lpips_values) / len(lpips_values)
        metrics["perceptual_metric_backend"] = f"piq-{getattr(piq, '__version__', 'unknown')}"
    except Exception as exc:
        metrics["perceptual_metric_error"] = f"{type(exc).__name__}: {exc}"
    return reconstructed, metrics

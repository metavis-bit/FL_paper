"""Export one exact client-batch gradient bundle for the two attack runners."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from flcore.attacks.gi_nas.bundle import build_victim_model, save_gradient_bundle
from flcore.attacks.gi_nas.official.utils.reconstructed import convert_relu_to_sigmoid


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(value)


def _parse_indices(value: str | None) -> Sequence[int] | None:
    if value is None:
        return None
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--indices must contain at least one integer")
    return indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _load_checkpoint(model: nn.Module, path: Path) -> None:
    value = _torch_load(path)
    if isinstance(value, nn.Module):
        state_dict = value.state_dict()
    elif isinstance(value, Mapping):
        state_dict = value.get("model_state_dict", value.get("state_dict", value))
    else:
        raise ValueError(
            "checkpoint must be a torch module or a state-dict-like mapping"
        )
    model.load_state_dict(state_dict, strict=True)


def _load_client_batch(
    path: Path,
    batch_size: int,
    sample_seed: int,
    explicit_indices: Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    archive = np.load(path, allow_pickle=True)
    if "data" not in archive:
        raise ValueError(f"client archive has no 'data' field: {path}")
    data = archive["data"].tolist()
    if not isinstance(data, dict) or "x" not in data or "y" not in data:
        raise ValueError("client archive 'data' must contain x and y")

    images = torch.as_tensor(np.asarray(data["x"]), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(data["y"]), dtype=torch.long)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected client images [N,3,H,W], got {tuple(images.shape)}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("client image and label counts differ")

    if explicit_indices is None:
        if batch_size < 2:
            raise ValueError("--batch-size must be at least 1")
        if images.shape[0] < batch_size:
            raise ValueError(
                f"client has {images.shape[0]} samples, fewer than batch size {batch_size}"
            )
        generator = torch.Generator().manual_seed(sample_seed)
        selected = torch.randperm(images.shape[0], generator=generator)[:batch_size]
    else:
        selected = torch.as_tensor(explicit_indices, dtype=torch.long)
        if selected.numel() < 1:
            raise ValueError("at least one explicit index is required")
        if int(selected.min()) < 0 or int(selected.max()) >= images.shape[0]:
            raise IndexError("an explicit sample index is outside the client archive")

    return images[selected], labels[selected], selected.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one exact client-batch gradient for GI-NAS and HF-GradInv"
    )
    parser.add_argument("--client-data", required=True, help="Client train .npz file")
    parser.add_argument("--output", required=True, help="Output gradient bundle (.pt)")
    parser.add_argument("--model", default="resnet", help="Victim model name")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--indices", help="Optional comma-separated sample indices")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        help="Trained victim checkpoint (.pth/.pt). Required for paper experiments.",
    )
    parser.add_argument("--round-index", type=int, default=None, help="Capture-round metadata")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--algorithm", default="FedAvg", help="Metadata only")
    parser.add_argument("--gradient-noise", type=float, default=0.0)
    parser.add_argument("--convert-relu", action="store_true")
    args = parser.parse_args()

    if args.gradient_noise < 0:
        raise ValueError("--gradient-noise must be non-negative")
    device = _device(args.device)
    images, labels, selected_indices = _load_client_batch(
        Path(args.client_data),
        args.batch_size,
        args.sample_seed,
        _parse_indices(args.indices),
    )
    images = images.to(device)
    labels = labels.to(device)

    torch.manual_seed(args.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.model_seed)
    model = build_victim_model(args.model, args.num_classes).to(device)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)
    model.eval()
    if args.convert_relu:
        convert_relu_to_sigmoid(model)
    criterion = nn.CrossEntropyLoss().to(device)
    model.zero_grad(set_to_none=True)
    loss = criterion(model(images), labels)
    gradients = [
        gradient.detach().clone()
        for gradient in torch.autograd.grad(loss, model.parameters())
    ]
    if args.gradient_noise > 0:
        noise_generator = torch.Generator(device=device).manual_seed(args.model_seed + 1)
        gradients = [
            gradient
            + torch.randn(
                gradient.shape,
                dtype=gradient.dtype,
                device=gradient.device,
                generator=noise_generator,
            )
            * args.gradient_noise
            for gradient in gradients
        ]

    output = save_gradient_bundle(
        args.output,
        model,
        model_name=args.model,
        num_classes=args.num_classes,
        gradients=gradients,
        labels=labels,
        normalized_images=images,
        metadata={
            "source": "standalone_exact_client_batch_gradient",
            "threat_model_id": "A_legacy_controlled_exact_batch_gradient",
            "observation_mode": "legacy_standalone_exact_gradient",
            "is_exact_gradient_at_saved_pre_model": True,
            "not_server_visible_upload": True,
            "capture_mode": "evaluation_model_before_local_update",
            "algorithm": args.algorithm,
            "client_data": str(Path(args.client_data)),
            "sample_indices": selected_indices,
            "sample_seed": args.sample_seed,
            "model_seed": args.model_seed,
            "model_initialization": "trained_checkpoint" if checkpoint_path else "random_seed",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
            "client_data_sha256": _sha256(Path(args.client_data).resolve()),
            "gradient_noise_std": args.gradient_noise,
            "relu_to_sigmoid": args.convert_relu,
            "round_index": args.round_index,
        },
    )
    print(f"Gradient bundle saved to: {output.resolve()}")
    print(f"Sample indices: {selected_indices}")
    print(f"Labels: {labels.detach().cpu().tolist()}")
    print(f"Cross-entropy loss: {float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()

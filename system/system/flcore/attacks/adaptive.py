"""One-step surrogate attack on Adap-CTA's server-visible model update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _distance(candidate: Sequence[torch.Tensor], target: Sequence[torch.Tensor]) -> torch.Tensor:
    value = target[0].new_zeros(())
    for lhs, rhs in zip(candidate, target):
        value = value + (lhs - rhs).pow(2).sum() / (rhs.pow(2).sum() + 1e-12)
    return value / max(len(target), 1)


def _total_variation(images: torch.Tensor) -> torch.Tensor:
    if images.shape[-1] < 2 or images.shape[-2] < 2:
        return images.new_zeros(())
    return (
        (images[..., 1:] - images[..., :-1]).abs().mean()
        + (images[..., 1:, :] - images[..., :-1, :]).abs().mean()
    )


def _cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_adaptive_observation(
    path: str | Path,
    model: nn.Module,
    sc: Sequence[torch.Tensor],
    delta_cc: Sequence[torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    num_batches: int,
    local_epochs: int,
    round_index: int,
    client_id: int,
    decision_l: int,
    num_classes: int = 10,
    model_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    """Save exactly the upload transcript; keep reference data evaluation-only."""

    if model_state_dict is None:
        raise ValueError("pre-update model state is required")
    if learning_rate <= 0 or num_batches <= 0 or local_epochs <= 0:
        raise ValueError("learning rate and local step counts must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_state = {name: value.detach().cpu().clone() for name, value in model_state_dict.items()}
    local_state = _cpu_state(model)
    model_update = [
        local_state[name] - pre_state[name] for name, _ in model.named_parameters()
    ]
    if len(model_update) != len(sc) or len(sc) != len(delta_cc):
        raise ValueError("model update, sc and delta_cc must have the same number of tensors")
    input_low = float(images.detach().min())
    input_high = float(images.detach().max())
    if input_high <= input_low:
        input_low, input_high = -1.0, 1.0
    payload = {
        "format": "fl-paper-adaptive-observation-bundle",
        "format_version": 2,
        "attack": {
            "model_state_dict": pre_state,
            "local_model_state_dict": local_state,
            "model_update": model_update,
            "sc": [value.detach().cpu().clone() for value in sc],
            "delta_cc": [value.detach().cpu().clone() for value in delta_cc],
            "decision_l": int(decision_l),
            "image_shape": tuple(int(value) for value in images.shape[1:]),
            "batch_size": int(images.shape[0]),
            "num_classes": int(num_classes),
            "learning_rate": float(learning_rate),
            "num_batches": int(num_batches),
            "local_epochs": int(local_epochs),
            "round_index": int(round_index),
            "client_id": int(client_id),
            "input_low": input_low,
            "input_high": input_high,
        },
        "evaluation_only": {
            "images": images.detach().cpu(),
            "labels": labels.detach().to(dtype=torch.long).cpu(),
        },
    }
    torch.save(payload, path)
    return path


def load_adaptive_observation(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if bundle.get("format") != "fl-paper-adaptive-observation-bundle":
        raise ValueError("not an Adap-CTA adaptive-observation bundle")
    if bundle.get("format_version") != 2:
        raise ValueError("old adaptive bundle contains a private raw gradient; capture it again")
    attack = bundle.get("attack", {})
    required = {"model_state_dict", "local_model_state_dict", "model_update", "sc", "delta_cc", "decision_l", "image_shape"}
    missing = sorted(required.difference(attack))
    if missing:
        raise ValueError(f"adaptive bundle is missing: {', '.join(missing)}")
    if len(attack["model_update"]) != len(attack["sc"]) or len(attack["sc"]) != len(attack["delta_cc"]):
        raise ValueError("model update, sc and delta_cc must have the same number of tensors")
    steps = int(attack["num_batches"]) * int(attack["local_epochs"])
    scale = steps * float(attack["learning_rate"])
    if scale <= 0:
        raise ValueError("invalid local step scale")
    for update, server_c, delta in zip(attack["model_update"], attack["sc"], attack["delta_cc"]):
        if not torch.allclose(-update / scale, server_c + delta, atol=1e-5, rtol=1e-4):
            raise ValueError("model update and delta_cc disagree with the training protocol")
    return bundle


@dataclass
class AdaptiveAttackConfig:
    iterations: int = 800
    learning_rate: float = 0.1
    restarts: int = 1
    tv_weight: float = 1e-4
    latent_weight: float = 1e-3
    seed: int = 42


class AdaptiveAttack:
    """Gradient inversion constrained by Adap-CTA's public update transcript."""

    def __init__(self, config: AdaptiveAttackConfig | None = None):
        self.config = config or AdaptiveAttackConfig()
        if self.config.iterations <= 0 or self.config.restarts <= 0:
            raise ValueError("iterations and restarts must be positive")

    def reconstruct(self, model: nn.Module, bundle: Mapping[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        attack = bundle["attack"]
        device = next(model.parameters()).device
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        steps = int(attack["num_batches"]) * int(attack["local_epochs"])
        scale = steps * float(attack["learning_rate"])
        target_update = [value.to(device=device, dtype=torch.float32) for value in attack["model_update"]]
        sc = [value.to(device=device, dtype=torch.float32) for value in attack["sc"]]
        delta_cc = [value.to(device=device, dtype=torch.float32) for value in attack["delta_cc"]]
        if len(parameters) != len(target_update):
            raise ValueError("observation/model parameter count mismatch")
        target_effective_gradient = [-update / scale for update in target_update]
        correction_start = max(0, len(parameters) - int(attack["decision_l"]))
        batch_size = int(attack["batch_size"])
        channels, height, width = (int(value) for value in attack["image_shape"])
        num_classes = int(attack.get("num_classes", bundle.get("num_classes", 10)))
        low, high = float(attack.get("input_low", -1.0)), float(attack.get("input_high", 1.0))
        low, high = min(low, high), max(high, low + 1e-3)

        last_bias = next((value for value in reversed(target_effective_gradient) if value.ndim == 1 and value.numel() == num_classes), None)
        if last_bias is None:
            label_init = torch.zeros(batch_size, num_classes, device=device)
        else:
            probabilities = (-last_bias).softmax(0).clamp_min(1e-6)
            label_init = probabilities.log().expand(batch_size, -1).clone()

        best_images, best_score, best_stats = None, math.inf, {}
        was_training = model.training
        model.eval()
        for restart in range(self.config.restarts):
            generator = torch.Generator(device=device).manual_seed(self.config.seed + restart)
            raw_images = nn.Parameter(torch.randn(
                batch_size, channels, height, width, device=device, generator=generator
            ) * 0.1)
            label_logits = nn.Parameter(label_init.clone())
            cc_scale = nn.Parameter(torch.zeros(len(parameters), device=device))
            optimizer = torch.optim.Adam(
                [raw_images, label_logits, cc_scale],
                lr=self.config.learning_rate,
            )
            for _ in range(self.config.iterations):
                optimizer.zero_grad(set_to_none=True)
                images = low + (raw_images.tanh() + 1.0) * 0.5 * (high - low)
                labels = label_logits.softmax(-1)
                logits = model(images)
                loss = -(labels * logits.log_softmax(-1)).sum(-1).mean()
                candidate_gradient = torch.autograd.grad(loss, parameters, create_graph=True)
                predicted_effective_gradient = []
                predicted_delta_cc = []
                for layer_index, (gradient, server_c) in enumerate(zip(candidate_gradient, sc)):
                    hidden_cc = cc_scale[layer_index].tanh() * target_effective_gradient[layer_index]
                    if layer_index >= correction_start:
                        predicted_effective_gradient.append(gradient + server_c - hidden_cc)
                        predicted_delta_cc.append(gradient - hidden_cc)
                    else:
                        predicted_effective_gradient.append(gradient)
                        predicted_delta_cc.append(gradient - server_c)
                update_loss = _distance(predicted_effective_gradient, target_effective_gradient)
                delta_loss = _distance(predicted_delta_cc, delta_cc)
                objective = (
                    update_loss + delta_loss
                    + self.config.tv_weight * _total_variation(images)
                    + self.config.latent_weight * cc_scale.tanh().pow(2).mean()
                )
                objective.backward()
                optimizer.step()

            with torch.enable_grad():
                images = low + (raw_images.tanh() + 1.0) * 0.5 * (high - low)
                labels = label_logits.softmax(-1)
                loss = -(labels * model(images).log_softmax(-1)).sum(-1).mean()
                candidate_gradient = torch.autograd.grad(loss, parameters, create_graph=False)
                predicted_effective_gradient = []
                predicted_delta_cc = []
                for layer_index, (gradient, server_c) in enumerate(zip(candidate_gradient, sc)):
                    hidden_cc = cc_scale[layer_index].tanh() * target_effective_gradient[layer_index]
                    if layer_index >= correction_start:
                        predicted_effective_gradient.append(gradient + server_c - hidden_cc)
                        predicted_delta_cc.append(gradient - hidden_cc)
                    else:
                        predicted_effective_gradient.append(gradient)
                        predicted_delta_cc.append(gradient - server_c)
                update_loss = float(_distance(predicted_effective_gradient, target_effective_gradient).cpu())
                delta_loss = float(_distance(predicted_delta_cc, delta_cc).cpu())
                score = update_loss + delta_loss
                if score < best_score:
                    best_score = score
                    best_images = images.detach().clone()
                    best_stats = {
                        "model_update_loss": update_loss,
                        "delta_cc_loss": delta_loss,
                        "delta_cc_consistency_error": float(_distance(
                            target_effective_gradient, [server_c + delta for server_c, delta in zip(sc, delta_cc)]
                        ).cpu()),
                        "corrected_tensor_count": float(max(0, len(parameters) - max(0, correction_start))),
                        "local_steps": float(steps),
                    }
        model.train(was_training)
        if best_images is None:
            raise RuntimeError("adaptive attack produced no reconstruction")
        return best_images, best_stats

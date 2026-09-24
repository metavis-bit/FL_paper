"""Project-facing implementation of HF-GradInv.

The original HF-GradInv attack (Ye et al., AAAI 2024) has two stages: it
infers a batch of labels from the last-layer gradient and then performs
stepwise gradient inversion, progressively adding gradients from the output
layer towards the input.  This module keeps that interface independent from
the upstream ``breaching`` experiment harness so it can consume the gradient
bundles produced by this project.

The implementation deliberately accepts ``labels=None``.  When labels are
provided they are intended for a controlled upper-bound/debug experiment;
the normal runner infers them from the observed gradients and uses the bundle
labels only for post-attack evaluation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HFGradInvConfig:
    """Configuration for the project adapter.

    ``iterations`` is the total number of optimization steps, distributed
    over the progressively growing gradient stages.  The official paper uses
    substantially larger values for its high-resolution experiments; smaller
    values are useful for wiring tests.
    """

    iterations: int = 3000
    stages: int = 4
    learning_rate: float = 0.1
    restarts: int = 1
    tv_weight: float = 1e-4
    gradient_dropout: float = 0.1
    signed_gradients: bool = False
    seed: int = 42
    report_every: int = 0
    auxiliary_weight: float = 0.25


def _total_variation(images: torch.Tensor) -> torch.Tensor:
    """An image prior used during the quality-improvement stage."""

    if images.shape[-1] < 2 or images.shape[-2] < 2:
        return images.new_zeros(())
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def _normalise_score(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float()
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values.clamp_min(0)
    total = values.sum()
    if float(total) <= 1e-12:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    return values / total


def _counts_to_labels(probabilities: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Convert a class-mass estimate into exactly ``batch_size`` labels."""

    if probabilities.numel() == 0:
        raise ValueError("cannot infer labels without a classifier gradient")
    expected = probabilities * int(batch_size)
    counts = torch.floor(expected).to(dtype=torch.long)
    remainder = int(batch_size - int(counts.sum()))
    if remainder > 0:
        fractional = expected - counts.to(expected.dtype)
        order = torch.argsort(fractional, descending=True)
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = torch.argsort(expected - counts.to(expected.dtype))
        for index in order:
            if remainder == 0:
                break
            if counts[index] > 0:
                counts[index] -= 1
                remainder += 1

    labels = torch.repeat_interleave(
        torch.arange(probabilities.numel(), device=probabilities.device), counts
    )
    if labels.numel() < batch_size:
        # This is only a numerical fallback for unusual gradients (for
        # example, a classifier with an empty or masked row).
        labels = torch.cat(
            [labels, torch.argmax(probabilities).repeat(batch_size - labels.numel())]
        )
    return labels[:batch_size].to(dtype=torch.long)


class HFGradInvAttack:
    """High-fidelity, stepwise gradient inversion for image classifiers."""

    def __init__(self, config: Optional[HFGradInvConfig] = None):
        self.config = config or HFGradInvConfig()
        if self.config.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.config.stages <= 0:
            raise ValueError("stages must be positive")
        if self.config.restarts <= 0:
            raise ValueError("restarts must be positive")
        if not 0 <= self.config.gradient_dropout < 1:
            raise ValueError("gradient_dropout must be in [0, 1)")

    @staticmethod
    def _trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    @staticmethod
    def _last_linear(model: nn.Module) -> Optional[nn.Linear]:
        last: Optional[nn.Linear] = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last = module
        return last

    @staticmethod
    def _capture_auxiliary_features(
        model: nn.Module,
        auxiliary_data: torch.Tensor,
        mean: Sequence[float],
        std: Sequence[float],
        image_shape: Sequence[int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Collect final-classifier features and their coefficient of variation."""

        linear = HFGradInvAttack._last_linear(model)
        if linear is None:
            return None
        if not isinstance(auxiliary_data, torch.Tensor) or auxiliary_data.ndim != 4:
            raise ValueError("auxiliary_data must have shape [N, C, H, W]")
        if auxiliary_data.shape[0] == 0:
            return None

        channels, height, width = (int(value) for value in image_shape)
        data = auxiliary_data.detach().to(next(model.parameters()).device).float()
        if tuple(data.shape[1:]) != (channels, height, width):
            data = F.interpolate(data, size=(height, width), mode="bilinear", align_corners=False)
        mean_tensor = data.new_tensor(mean).view(1, -1, 1, 1)
        std_tensor = data.new_tensor(std).view(1, -1, 1, 1)
        data = (data - mean_tensor) / std_tensor

        captured: Dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            if inputs:
                captured["features"] = inputs[0].detach()

        handle = linear.register_forward_pre_hook(hook)
        previous_mode = model.training
        model.eval()
        try:
            with torch.no_grad():
                # A single forward keeps the method simple and is adequate for
                # the small public auxiliary sets normally used by HF-GradInv.
                output = model(data)
                captured["probabilities"] = F.softmax(output, dim=1).mean(dim=0)
        finally:
            handle.remove()
            model.train(previous_mode)

        features = captured.get("features")
        if features is None:
            return None
        features = features.flatten(1).float()
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0, unbiased=False)
        cv = feature_std / feature_mean.abs().clamp_min(1e-6)
        cv = torch.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "mean": feature_mean,
            "cv": cv,
            "cv_mean": cv.mean(),
            "probabilities": captured["probabilities"],
        }

    @staticmethod
    def _infer_labels(
        model: nn.Module,
        target_gradient: Sequence[torch.Tensor],
        batch_size: int,
        auxiliary_features: Optional[Dict[str, torch.Tensor]],
        auxiliary_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Infer a label multiset from the last classifier gradient.

        The negative row mass is the standard analytical initialization for
        cross-entropy gradients.  When auxiliary features are available, a
        second score based on the final-layer feature mean is blended in; the
        feature coefficient of variation is reported for reproducibility.
        """

        parameters = list(model.parameters())
        gradient_by_id = {id(parameter): gradient for parameter, gradient in zip(parameters, target_gradient)}
        linear = HFGradInvAttack._last_linear(model)
        if linear is None:
            raise ValueError("HF-GradInv requires a final nn.Linear classifier")
        weight_gradient = gradient_by_id.get(id(linear.weight))
        bias_gradient = gradient_by_id.get(id(linear.bias)) if linear.bias is not None else None
        if weight_gradient is None:
            raise ValueError("target gradient does not contain the final classifier weight")

        if bias_gradient is not None and bias_gradient.ndim == 1:
            primary = (-bias_gradient).clamp_min(0)
        else:
            primary = (-weight_gradient).flatten(1).mean(dim=1).clamp_min(0)

        score = _normalise_score(primary)
        inference_mode = "final_classifier_negative_mass"
        stats: Dict[str, object] = {
            "inference_mode": inference_mode,
            "class_probability": score.detach().cpu().tolist(),
        }

        if auxiliary_features is not None:
            feature_mean = auxiliary_features["mean"].to(weight_gradient.device)
            auxiliary_score = None
            # For cross-entropy, bias_gradient ~= mean(model_probability) -
            # label_histogram.  Estimating mean(model_probability) from public
            # auxiliary images makes repeated labels recoverable instead of
            # merely identifying the class with the largest negative row.
            auxiliary_probabilities = auxiliary_features.get("probabilities")
            if (
                bias_gradient is not None
                and auxiliary_probabilities is not None
                and auxiliary_probabilities.numel() == bias_gradient.numel()
            ):
                estimated_histogram = (
                    auxiliary_probabilities.to(weight_gradient.device) - bias_gradient
                ).clamp_min(0)
                if float(estimated_histogram.sum()) > 1e-8:
                    auxiliary_score = _normalise_score(estimated_histogram)
            if auxiliary_score is None and feature_mean.numel() == weight_gradient.shape[1]:
                auxiliary_score = _normalise_score(
                    (-weight_gradient * feature_mean.view(1, -1)).sum(dim=1).clamp_min(0)
                )
            if auxiliary_score is not None:
                alpha = float(max(0.0, min(1.0, auxiliary_weight)))
                score = _normalise_score((1.0 - alpha) * score + alpha * auxiliary_score)
                inference_mode = "cv_assisted_final_classifier_inference"
                stats.update(
                    {
                        "inference_mode": inference_mode,
                        "auxiliary_weight": alpha,
                        "auxiliary_cv_mean": float(auxiliary_features["cv_mean"].detach().cpu()),
                        "class_probability": score.detach().cpu().tolist(),
                    }
                )

        labels = _counts_to_labels(score, batch_size)
        stats["inferred_labels"] = labels.detach().cpu().tolist()
        return labels, stats

    @staticmethod
    def _gradient_stages(num_gradients: int, stages: int) -> List[List[int]]:
        if num_gradients <= 0:
            raise ValueError("target_gradient cannot be empty")
        stage_count = min(int(stages), num_gradients)
        order = list(range(num_gradients - 1, -1, -1))
        result: List[List[int]] = []
        for stage in range(stage_count):
            end = max(1, math.ceil((stage + 1) * num_gradients / stage_count))
            result.append(sorted(order[:end]))
        # The final stage must always contain every parameter gradient.
        result[-1] = list(range(num_gradients))
        return result

    def _matching_loss(
        self,
        candidate: Sequence[torch.Tensor],
        target: Sequence[torch.Tensor],
        indices: Sequence[int],
        dropout: float,
    ) -> Tuple[torch.Tensor, int]:
        losses: List[torch.Tensor] = []
        kept = 0
        for index in indices:
            if dropout > 0 and torch.rand((), device=target[index].device) < dropout:
                continue
            lhs = candidate[index].reshape(-1)
            rhs = target[index].reshape(-1)
            losses.append(1.0 - F.cosine_similarity(lhs, rhs, dim=0, eps=1e-8))
            kept += 1
        if not losses:
            index = int(indices[-1])
            lhs = candidate[index].reshape(-1)
            rhs = target[index].reshape(-1)
            losses.append(1.0 - F.cosine_similarity(lhs, rhs, dim=0, eps=1e-8))
            kept = 1
        return torch.stack(losses).mean(), kept

    def _reconstruct_once(
        self,
        victim_model: nn.Module,
        target_gradient: Sequence[torch.Tensor],
        labels: torch.Tensor,
        image_shape: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        seed: int,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        device = next(victim_model.parameters()).device
        channels, height, width = (int(value) for value in image_shape)
        mean_tensor = torch.as_tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        std_tensor = torch.as_tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        lower = (-mean_tensor / std_tensor).expand(1, channels, height, width)
        upper = ((1.0 - mean_tensor) / std_tensor).expand(1, channels, height, width)

        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        # Work in normalized image space, as the client gradient was computed.
        dummy = torch.empty((labels.numel(), channels, height, width), device=device).uniform_(-1, 1)
        dummy = torch.max(torch.min(dummy, upper), lower).detach().requires_grad_(True)

        optimizer = torch.optim.Adam([dummy], lr=float(self.config.learning_rate))
        model_parameters = self._trainable_parameters(victim_model)
        criterion = nn.CrossEntropyLoss().to(device)
        stages = self._gradient_stages(len(target_gradient), self.config.stages)
        iterations_left = int(self.config.iterations)
        loss_history: List[Dict[str, float]] = []
        total_kept = 0

        for stage_index, indices in enumerate(stages):
            remaining_stages = len(stages) - stage_index
            stage_iterations = max(1, math.ceil(iterations_left / remaining_stages))
            iterations_left -= stage_iterations
            # The final quality stage uses all gradients and a milder dropout.
            dropout = 0.0 if stage_index == len(stages) - 1 else float(self.config.gradient_dropout)
            for iteration in range(stage_iterations):
                optimizer.zero_grad(set_to_none=True)
                victim_model.zero_grad(set_to_none=True)
                output = victim_model(dummy)
                loss = criterion(output, labels)
                candidate_gradient = torch.autograd.grad(
                    loss, model_parameters, create_graph=True, retain_graph=True
                )
                gradient_loss, kept = self._matching_loss(
                    candidate_gradient, target_gradient, indices, dropout
                )
                reconstructed_01 = (dummy * std_tensor + mean_tensor).clamp(0, 1)
                objective = gradient_loss + float(self.config.tv_weight) * _total_variation(
                    reconstructed_01
                )
                objective.backward()
                if self.config.signed_gradients and dummy.grad is not None:
                    dummy.grad.sign_()
                optimizer.step()
                with torch.no_grad():
                    dummy.clamp_(min=lower, max=upper)

                total_kept += kept
                should_report = (
                    self.config.report_every > 0
                    and (iteration + 1) % self.config.report_every == 0
                ) or iteration == 0 or iteration + 1 == stage_iterations
                if should_report:
                    loss_history.append(
                        {
                            "stage": float(stage_index + 1),
                            "iteration": float(iteration + 1),
                            "gradient_loss": float(gradient_loss.detach().cpu()),
                            "objective": float(objective.detach().cpu()),
                        }
                    )

        with torch.no_grad():
            reconstructed = (dummy * std_tensor + mean_tensor).clamp(0, 1).detach().cpu()
        stats: Dict[str, object] = {
            "loss_history": loss_history,
            "stages": [list(indices) for indices in stages],
            "iterations": int(self.config.iterations),
            "gradient_tensors_used": int(total_kept),
        }
        return reconstructed, stats

    def reconstruct(
        self,
        victim_model: nn.Module,
        target_gradient: Sequence[torch.Tensor],
        image_shape: Sequence[int],
        normalization_mean: Sequence[float] = (0.5, 0.5, 0.5),
        normalization_std: Sequence[float] = (0.5, 0.5, 0.5),
        labels: Optional[torch.Tensor] = None,
        auxiliary_data: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Reconstruct an image batch in ``[0, 1]``.

        Parameters
        ----------
        labels:
            Optional known labels for an upper-bound/debug run.  Leave it as
            ``None`` for the paper's label-inference setting.
        auxiliary_data:
            Optional public, disjoint auxiliary images in ``[0, 1]``.  Their
            labels are not needed; only final-layer feature statistics are
            used for the CV-assisted initialization.
        """

        if len(image_shape) != 3 or int(image_shape[0]) != 3:
            raise ValueError("HF-GradInv currently supports three-channel images")
        model_parameters = self._trainable_parameters(victim_model)
        if len(target_gradient) != len(model_parameters):
            raise ValueError(
                f"target gradient/model mismatch: {len(target_gradient)} vs {len(model_parameters)}"
            )
        device = next(victim_model.parameters()).device
        target = [gradient.detach().to(device=device) for gradient in target_gradient]
        for gradient, parameter in zip(target, model_parameters):
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise ValueError("target gradient tensor shapes do not match the victim model")

        if labels is not None:
            labels = labels.detach().to(device=device, dtype=torch.long).flatten()
            batch_size = int(labels.numel())
        else:
            batch_size = int(batch_size or 0)
        if batch_size <= 0:
            raise ValueError(
                "batch size is required when labels are hidden; pass batch_size explicitly "
                "or use HFGradInvAttack.reconstruct_from_bundle"
            )

        previous_mode = victim_model.training
        victim_model.eval()
        started = time.time()
        aux_stats = None
        if auxiliary_data is not None:
            aux_stats = self._capture_auxiliary_features(
                victim_model,
                auxiliary_data,
                normalization_mean,
                normalization_std,
                image_shape,
            )

        if labels is None:
            labels, label_stats = self._infer_labels(
                victim_model,
                target,
                batch_size,
                aux_stats,
                self.config.auxiliary_weight,
            )
        else:
            label_stats = {
                "inference_mode": "provided_labels",
                "inferred_labels": labels.detach().cpu().tolist(),
            }

        best_reconstruction: Optional[torch.Tensor] = None
        best_stats: Optional[Dict[str, object]] = None
        best_score = math.inf
        for restart in range(int(self.config.restarts)):
            reconstruction, run_stats = self._reconstruct_once(
                victim_model,
                target,
                labels,
                image_shape,
                normalization_mean,
                normalization_std,
                self.config.seed + restart,
            )
            final_loss = run_stats["loss_history"][-1]["objective"] if run_stats["loss_history"] else math.inf
            if float(final_loss) < best_score:
                best_score = float(final_loss)
                best_reconstruction = reconstruction
                best_stats = run_stats

        victim_model.zero_grad(set_to_none=True)
        victim_model.train(previous_mode)
        if best_reconstruction is None or best_stats is None:
            raise RuntimeError("HF-GradInv did not produce a reconstruction")

        stats: Dict[str, object] = {
            "method": "HF-GradInv",
            "elapsed_seconds": time.time() - started,
            "batch_size": batch_size,
            "restarts": int(self.config.restarts),
            "best_objective": best_score,
            "auxiliary_data_used": aux_stats is not None,
            "label_inference": label_stats,
            **best_stats,
        }
        return best_reconstruction, stats

    def reconstruct_from_bundle(
        self,
        victim_model: nn.Module,
        bundle: Dict[str, object],
        auxiliary_data: Optional[torch.Tensor] = None,
        use_ground_truth_labels: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Convenience wrapper for this project's serialized snapshots."""

        target_gradient = list(bundle["target_gradient"])
        batch_size = int(bundle["labels"].shape[0])
        if use_ground_truth_labels:
            labels = bundle["labels"]
        else:
            labels = None
        return self.reconstruct(
            victim_model,
            target_gradient,
            bundle["image_shape"],
            bundle["normalization_mean"],
            bundle["normalization_std"],
            labels=labels,
            auxiliary_data=auxiliary_data,
            batch_size=batch_size,
        )


# Short alias matching the paper/method name and making imports convenient for
# experiment scripts that use ``HFGradInv`` rather than the longer class name.
HFGradInv = HFGradInvAttack

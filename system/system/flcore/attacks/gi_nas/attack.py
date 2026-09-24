"""Adapter around the official GI-NAS architecture search and optimization.

The generator architecture, search space, gradient cosine loss, signed Adam
updates, and learning-rate schedule follow the official implementation.  This
adapter makes the attack usable with a model/gradient snapshot exported by the
federated-learning clients without importing the official notebook-oriented
training harness.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .official.model.cross_skip import skip
from .official.utils.reconstructed import loss_cosine_similarity, total_variation


@dataclass
class GINASConfig:
    """Configuration matching the official CIFAR-10 GI-NAS experiment."""

    search_size: int = 5000
    iterations: int = 30000
    learning_rate: float = 1e-3
    tv_weight: float = 0.0
    signed_gradients: bool = True
    learning_rate_decay: bool = True
    generator_channels: int = 128
    skip_channels: int = 4
    model_index_modulus: int = 300
    seed: int = 42
    report_every: int = 0
    verbose: bool = True
    search_space_path: Optional[str] = None


class GINASAttack:
    """Run the two-stage GI-NAS attack against a known victim model."""

    def __init__(self, config: Optional[GINASConfig] = None):
        self.config = config or GINASConfig()
        if self.config.search_size <= 0:
            raise ValueError("search_size must be positive")
        if self.config.iterations <= 0:
            raise ValueError("iterations must be positive")

    @staticmethod
    def _normalize(
        images: torch.Tensor,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> torch.Tensor:
        mean_tensor = images.new_tensor(mean).view(1, -1, 1, 1)
        std_tensor = images.new_tensor(std).view(1, -1, 1, 1)
        return (images - mean_tensor) / std_tensor

    @staticmethod
    def _cosine_loss(
        target_gradient: Sequence[torch.Tensor],
        candidate_gradient: Sequence[torch.Tensor],
        candidate_images: torch.Tensor,
        tv_weight: float,
    ) -> torch.Tensor:
        loss = loss_cosine_similarity(target_gradient, candidate_gradient)
        if tv_weight:
            loss = loss + tv_weight * total_variation(candidate_images)
        return loss

    def _read_search_space(self) -> List[np.ndarray]:
        path = (
            Path(self.config.search_space_path)
            if self.config.search_space_path
            else Path(__file__).parent / "official" / "model_search_space.txt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"GI-NAS search space was not found: {path}")

        candidates: List[np.ndarray] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value:
                    continue
                if len(value) != 25 or any(bit not in "01" for bit in value):
                    raise ValueError(
                        f"invalid GI-NAS architecture at {path}:{line_number}"
                    )
                candidates.append(np.asarray([int(bit) for bit in value]).reshape(5, 5))
                if len(candidates) >= self.config.search_size:
                    break

        if not candidates:
            raise ValueError(f"GI-NAS search space is empty: {path}")
        return candidates

    def _make_generator(self, model_index: int, skip_index: np.ndarray) -> nn.Module:
        channels = self.config.generator_channels
        return skip(
            model_index=model_index % self.config.model_index_modulus,
            skip_index=skip_index,
            num_input_channels=3,
            num_output_channels=3,
            num_channels_down=[channels] * 5,
            num_channels_up=[channels] * 5,
            num_channels_skip=[self.config.skip_channels] * 5,
            upsample_mode="bilinear",
            downsample_mode="stride",
            need_sigmoid=True,
            need_bias=True,
            pad="constant",
            act_fun="LeakyReLU",
        )

    def reconstruct(
        self,
        victim_model: nn.Module,
        target_gradient: Sequence[torch.Tensor],
        labels: torch.Tensor,
        image_shape: Sequence[int],
        normalization_mean: Sequence[float] = (0.5, 0.5, 0.5),
        normalization_std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Reconstruct a batch and return images in the unnormalized [0, 1] range."""

        if len(image_shape) != 3 or int(image_shape[0]) != 3:
            raise ValueError("the official GI-NAS generator requires 3-channel images")
        if labels.ndim != 1 or labels.numel() < 2:
            raise ValueError(
                "GI-NAS requires batch size >= 2 because the official generator's "
                "deepest BatchNorm receives a 1x1 feature map"
            )

        try:
            device = next(victim_model.parameters()).device
        except StopIteration as exc:
            raise ValueError("victim_model has no parameters") from exc

        model_parameters = [parameter for parameter in victim_model.parameters() if parameter.requires_grad]
        if len(target_gradient) != len(model_parameters):
            raise ValueError(
                "target gradient/model mismatch: "
                f"{len(target_gradient)} gradient tensors for {len(model_parameters)} parameters"
            )

        labels = labels.detach().to(device=device, dtype=torch.long)
        target_gradient = [gradient.detach().to(device) for gradient in target_gradient]
        expected_shapes = [tuple(parameter.shape) for parameter in model_parameters]
        gradient_shapes = [tuple(gradient.shape) for gradient in target_gradient]
        if gradient_shapes != expected_shapes:
            raise ValueError("target gradient tensor shapes do not match the victim model")

        batch_size = int(labels.numel())
        channels, height, width = (int(value) for value in image_shape)
        criterion = nn.CrossEntropyLoss().to(device)
        previous_training_mode = victim_model.training
        victim_model.eval()
        victim_model.zero_grad(set_to_none=True)

        torch.manual_seed(self.config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        noise = torch.randn(batch_size, channels, height, width, device=device)

        start_time = time.time()
        candidates = self._read_search_space()
        best_generator: Optional[nn.Module] = None
        best_metric = math.inf
        best_model_index = -1
        failed_candidates = 0

        try:
            for model_index, skip_index in enumerate(candidates):
                trial_generator: Optional[nn.Module] = None
                try:
                    trial_generator = self._make_generator(model_index, skip_index).to(device)
                    trial_images = trial_generator(noise)
                    victim_input = self._normalize(
                        trial_images, normalization_mean, normalization_std
                    )
                    dummy_loss = criterion(victim_model(victim_input), labels)
                    candidate_gradient = torch.autograd.grad(
                        dummy_loss, model_parameters, create_graph=False
                    )
                    search_metric = float(
                        loss_cosine_similarity(target_gradient, candidate_gradient)
                        .detach()
                        .cpu()
                    )
                    if math.isfinite(search_metric) and search_metric < best_metric:
                        best_metric = search_metric
                        best_model_index = model_index
                        best_generator = copy.deepcopy(trial_generator)
                except (RuntimeError, ValueError) as exc:
                    failed_candidates += 1
                    if self.config.verbose:
                        print(f"GI-NAS search candidate {model_index} failed: {exc}")
                finally:
                    del trial_generator

                if self.config.verbose and (
                    model_index == 0
                    or (model_index + 1) % 100 == 0
                    or model_index + 1 == len(candidates)
                ):
                    print(
                        "GI-NAS search "
                        f"{model_index + 1}/{len(candidates)}, "
                        f"best={best_model_index}, loss={best_metric:.6f}"
                    )

            if best_generator is None:
                raise RuntimeError("all GI-NAS architecture candidates failed")

            generator = best_generator.to(device)
            optimizer = torch.optim.Adam(
                generator.parameters(), lr=self.config.learning_rate
            )
            scheduler = None
            if self.config.learning_rate_decay:
                milestones = sorted(
                    {
                        max(1, int(self.config.iterations * 3 / 8)),
                        max(1, int(self.config.iterations * 5 / 8)),
                        max(1, int(self.config.iterations * 7 / 8)),
                    }
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=0.1
                )

            report_every = self.config.report_every
            if report_every <= 0:
                report_every = max(1, self.config.iterations // 250)
            loss_history: List[Dict[str, float]] = []
            reconstructed = None

            for iteration in range(self.config.iterations):
                optimizer.zero_grad(set_to_none=True)
                reconstructed = generator(noise)
                victim_input = self._normalize(
                    reconstructed, normalization_mean, normalization_std
                )
                dummy_loss = criterion(victim_model(victim_input), labels)
                candidate_gradient = torch.autograd.grad(
                    dummy_loss, model_parameters, create_graph=True
                )
                gradient_loss = self._cosine_loss(
                    target_gradient,
                    candidate_gradient,
                    reconstructed,
                    self.config.tv_weight,
                )
                gradient_loss.backward()

                if self.config.signed_gradients:
                    for parameter in generator.parameters():
                        if parameter.grad is not None:
                            parameter.grad.sign_()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                should_report = (
                    iteration == 0
                    or (iteration + 1) % report_every == 0
                    or iteration + 1 == self.config.iterations
                )
                if should_report:
                    current_loss = float(gradient_loss.detach().cpu())
                    loss_history.append(
                        {"iteration": iteration + 1, "gradient_loss": current_loss}
                    )
                    if self.config.verbose:
                        print(
                            f"GI-NAS optimize {iteration + 1}/{self.config.iterations}, "
                            f"loss={current_loss:.6f}"
                        )

            if reconstructed is None:
                raise RuntimeError("GI-NAS did not produce a reconstruction")

            with torch.no_grad():
                reconstructed = generator(noise).detach().clamp(0, 1).cpu()
            stats: Dict[str, object] = {
                "method": "GI-NAS",
                "best_model_index": best_model_index,
                "best_search_metric": best_metric,
                "searched_candidates": len(candidates),
                "failed_candidates": failed_candidates,
                "iterations": self.config.iterations,
                "loss_history": loss_history,
                "elapsed_seconds": time.time() - start_time,
            }
            return reconstructed, stats
        finally:
            victim_model.zero_grad(set_to_none=True)
            victim_model.train(previous_training_mode)

"""Dependency-free client adapters for the paper privacy baselines.

The official FedCEO and AdapLDP-FL projects use different training stacks. The
adapters keep this repository's local-SGD path and add the method-specific
update mechanisms at its boundary, without requiring Opacus.
"""

import torch

from flcore.clients.clientavg import clientAVG


def _update_norm(model, reference):
    """Return the L2 norm of a model update without autograd history."""
    squared = None
    for parameter, old in zip(model.parameters(), reference):
        value = (parameter.detach() - old).float()
        term = torch.sum(value * value)
        squared = term if squared is None else squared + term
    return torch.sqrt(squared) if squared is not None else torch.tensor(0.0)


def _clip_and_noise(model, reference, clip, noise_std, samples):
    """Clip a client update and add Gaussian noise in update space.

    This is a dependency-free update-space approximation, not a claim of
    bitwise equivalence to Opacus per-example DP-SGD.
    """
    clip = max(float(clip), 1e-12)
    norm = _update_norm(model, reference)
    scale = min(1.0, clip / (float(norm.item()) + 1e-12))
    noise_std = max(float(noise_std), 0.0) * clip / max(float(samples), 1.0) ** 0.5
    with torch.no_grad():
        for parameter, old in zip(model.parameters(), reference):
            delta = (parameter - old) * scale
            if noise_std:
                delta = delta + torch.randn_like(delta) * noise_std
            parameter.copy_(old + delta)
    return float(norm.item()), float(scale)


class clientFedCEO(clientAVG):
    """FedCEO-compatible client: clipped/noised local update."""

    def set_parameters(self, model):
        super().set_parameters(model)
        self._fedceo_reference = [p.detach().clone() for p in self.model.parameters()]

    def train(self):
        super().train()
        reference = getattr(self, "_fedceo_reference", None)
        if reference is None:
            return
        noise_multiplier = getattr(self.args, "fedceo_noise_multiplier", None)
        if noise_multiplier is None:
            noise_multiplier = getattr(self.args, "noise_multiplier", 0.0)
        clip = getattr(self.args, "fedceo_max_grad_norm", 1.0)
        norm, clip_scale = _clip_and_noise(
            self.model, reference, clip, noise_multiplier, self.train_samples
        )
        self.fedceo_update_norm = norm
        self.fedceo_clip_scale = clip_scale


class clientAdapLDP(clientAVG):
    """AdapLDP-FL-compatible client with adaptive client noise scaling."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.scaler_c_i = 1.0

    def set_parameters(self, model):
        super().set_parameters(model)
        self._adapldp_reference = [p.detach().clone() for p in self.model.parameters()]

    def train(self):
        super().train()
        reference = getattr(self, "_adapldp_reference", None)
        if reference is None:
            return

        clip = max(float(getattr(self.args, "adapldp_clip", 1.0)), 1e-12)
        norm = _update_norm(self.model, reference)
        target_scale = max(float(norm.item()) / clip, 1e-3)
        self.scaler_c_i = min(
            float(getattr(self.args, "adapldp_max_scaler", 4.0)),
            max(
                float(getattr(self.args, "adapldp_min_scaler", 0.25)),
                0.9 * self.scaler_c_i + 0.1 * target_scale,
            ),
        )
        multiplier = getattr(self.args, "adapldp_noise_multiplier", None)
        if multiplier is None:
            multiplier = getattr(self.args, "noise_multiplier", 0.0)
        noise_std = max(float(multiplier), 0.0) * clip * self.scaler_c_i / max(
            float(self.train_samples), 1.0
        ) ** 0.5
        mechanism = str(getattr(self.args, "adapldp_mechanism", "gaussian")).lower()
        with torch.no_grad():
            scale = min(1.0, clip / (float(norm.item()) + 1e-12))
            for parameter, old in zip(self.model.parameters(), reference):
                delta = (parameter - old) * scale
                if noise_std:
                    if mechanism == "laplace":
                        noise = torch.distributions.Laplace(
                            torch.zeros_like(delta), torch.full_like(delta, noise_std)
                        ).sample()
                    else:
                        noise = torch.randn_like(delta) * noise_std
                    delta = delta + noise
                parameter.copy_(old + delta)
        self.adapldp_update_norm = float(norm.item())
        self.adapldp_noise_scale = float(self.scaler_c_i)

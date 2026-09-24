"""Minimal FedCEO and AdapLDP-FL adapters for comparable baseline runs."""

import torch

from flcore.clients.client_privacy import clientAdapLDP, clientFedCEO
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbase import Server


def add_privacy_baseline_args(parser):
    """Register only the knobs used by the two replacement baselines."""
    parser.add_argument("--fedceo-noise-multiplier", type=float, default=None)
    parser.add_argument("--fedceo-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fedceo-lamb", type=float, default=0.6)
    parser.add_argument("--fedceo-r", type=float, default=1.04)
    parser.add_argument("--fedceo-interval", type=int, default=10)
    parser.add_argument("--adapldp-noise-multiplier", type=float, default=None)
    parser.add_argument("--adapldp-clip", type=float, default=1.0)
    parser.add_argument(
        "--adapldp-mechanism", choices=("gaussian", "laplace"), default="gaussian"
    )
    parser.add_argument("--adapldp-direction-rate", type=float, default=0.5)


def _weighted_mean(rows, weights):
    weight = torch.as_tensor(weights, device=rows.device, dtype=rows.dtype)
    weight = weight / weight.sum().clamp_min(1e-12)
    return (rows * weight[:, None]).sum(dim=0)


def _low_rank_smooth(rows, lamb, rank_growth):
    """Shrink client-space singular values of stacked client models.

    The Gram matrix is only ``clients x clients``; this avoids a
    ``parameters x parameters`` matrix for ResNet-sized models.
    """
    if rows.shape[0] < 2:
        return rows
    gram = rows @ rows.t()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    singular = eigenvalues.clamp_min(0).sqrt()
    threshold = max(float(lamb), 0.0) * max(float(rank_growth), 1.0)
    factors = torch.where(
        singular > 1e-12,
        (singular - threshold).clamp_min(0.0) / singular.clamp_min(1e-12),
        torch.zeros_like(singular),
    )
    coefficients = eigenvectors.t() @ rows
    return eigenvectors @ (coefficients * factors[:, None])


class FedCEO(FedAvg):
    """FedCEO-compatible low-rank semantic smoothing after local updates."""

    def __init__(self, args, times):
        self._fedceo_round = 0
        super().__init__(args, times)
        self.baseline_metadata = {
            "implementation": "FedCEO-compatible adapter",
            "aggregation": "client-space low-rank singular-value shrinkage",
            "upstream": "https://github.com/6lyc/FedCEO_Collaborate-with-Each-Other",
            "noise_multiplier": getattr(args, "fedceo_noise_multiplier", None)
            if getattr(args, "fedceo_noise_multiplier", None) is not None
            else getattr(args, "noise_multiplier", 0.0),
            "max_grad_norm": getattr(args, "fedceo_max_grad_norm", 1.0),
            "lamb": getattr(args, "fedceo_lamb", 0.6),
            "r": getattr(args, "fedceo_r", 1.04),
            "interval": getattr(args, "fedceo_interval", 10),
        }

    def set_clients(self, _client_obj=None):
        # FedAvg.__init__ calls set_clients(clientAVG); select the adapter
        # without duplicating the parent constructor or training loop.
        return Server.set_clients(self, clientFedCEO)

    def aggregate_parameters(self):
        assert self.uploaded_models
        self._fedceo_round += 1
        names = [name for name, _ in self.global_model.named_parameters()]
        client_parameters = [dict(model.named_parameters()) for model in self.uploaded_models]
        vectors = [
            torch.cat([parameters[name].detach().reshape(-1) for name in names])
            for parameters in client_parameters
        ]
        rows = torch.stack(vectors)
        interval = max(int(getattr(self.args, "fedceo_interval", 10)), 1)
        smoothing = bool(getattr(self.args, "fedceo_enable_smoothing", True))
        if smoothing and self._fedceo_round % interval == 0 and rows.shape[0] > 1:
            rows = _low_rank_smooth(
                rows,
                getattr(self.args, "fedceo_lamb", 0.6),
                getattr(self.args, "fedceo_r", 1.04) ** (self._fedceo_round // interval),
            )
        aggregate = _weighted_mean(rows, self.uploaded_weights)
        offset = 0
        with torch.no_grad():
            for name, parameter in self.global_model.named_parameters():
                size = parameter.numel()
                parameter.copy_(aggregate[offset : offset + size].view_as(parameter))
                offset += size


class AdapLDPFL(FedAvg):
    """AdapLDP-FL-compatible adaptive noise and direction correction."""

    def __init__(self, args, times):
        self._last_global_delta = None
        super().__init__(args, times)
        self.baseline_metadata = {
            "implementation": "AdapLDP-FL-compatible adapter",
            "aggregation": "adaptive update noise and global-direction correction",
            "upstream": "https://github.com/liyan2015/AdapLDP-FL",
            "noise_multiplier": getattr(args, "adapldp_noise_multiplier", None)
            if getattr(args, "adapldp_noise_multiplier", None) is not None
            else getattr(args, "noise_multiplier", 0.0),
            "clip": getattr(args, "adapldp_clip", 1.0),
            "mechanism": getattr(args, "adapldp_mechanism", "gaussian"),
            "direction_rate": getattr(args, "adapldp_direction_rate", 0.5),
        }

    def set_clients(self, _client_obj=None):
        return Server.set_clients(self, clientAdapLDP)

    def aggregate_parameters(self):
        assert self.uploaded_models
        old_parameters = [p.detach().clone() for p in self.global_model.parameters()]
        reference = self._last_global_delta
        direction_rate = min(
            max(float(getattr(self.args, "adapldp_direction_rate", 0.5)), 0.0), 1.0
        )
        local_updates = []
        for model in self.uploaded_models:
            updates = [
                (new.detach() - old).reshape(-1)
                for new, old in zip(model.parameters(), old_parameters)
            ]
            local_updates.append(torch.cat(updates))
        rows = torch.stack(local_updates)
        if reference is not None and direction_rate:
            ref = reference.to(rows.device)
            ref_norm = torch.dot(ref, ref).clamp_min(1e-12)
            corrected = []
            for row in rows:
                coefficient = torch.dot(row, ref) / ref_norm
                projection = coefficient * ref
                corrected.append((1.0 - direction_rate) * row + direction_rate * projection)
            rows = torch.stack(corrected)
        aggregate_delta = _weighted_mean(rows, self.uploaded_weights)
        offset = 0
        with torch.no_grad():
            for parameter, old in zip(self.global_model.parameters(), old_parameters):
                size = parameter.numel()
                parameter.copy_(old + aggregate_delta[offset : offset + size].view_as(parameter))
                offset += size
        self._last_global_delta = aggregate_delta.detach().clone()


if __name__ == "__main__":
    # The FedCEO smoothing step must affect the aggregate even for equal-size clients.
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
    smoothed = _low_rank_smooth(rows, 0.1, 1.0)
    assert not torch.allclose(
        _weighted_mean(rows, [1, 1, 1]),
        _weighted_mean(smoothed, [1, 1, 1]),
    )

"""Mechanisms for image reconstruction from parameter gradients."""

import torch
import time

from torch import nn
from typing import Dict, List, Tuple, Any, Union, Callable
from .inversefed.metrics import total_variation as TV
from .inversefed.metrics import InceptionScore
from .inversefed.medianfilt import MedianPool2d

DEFAULT_CONFIG = {
    "signed": True,
    "boxed": True,
    "cost_fn": 'sim',
    "indices": 'def',
    "weights": 'equal',
    "lr": 0.1,
    "optim": 'sgd',
    "restarts": 5,
    "max_iterations": 2000,
    "init": 'randn',
    "filter": 'none',
    "lr_decay": True,
    "scoring_choice": 'loss'
}

TOTAL_VARIATIONS = {
    "ResNet18": 1e-3,
    "ResNet34": 1e-6,
    "ResNet50": 1e-6,
    "LeNet5": 1e-6,
    "ShuffleNetV2": 0.1,
    "GoogleNet": 0.1,
    "AlexNet": 1e-2,
    "MobileNetV2": 1e-6,
    "VGG16": 1e-6,
}


def _validate_config(config: Dict[str, Any]):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor:
    """Instantiate a reconstruction algorithm."""

    def __init__(
            self, model_name: str, server_model: nn.Module,
            device: torch.device, mean_std=(0.0, 1.0),
            config: Dict[str, Any] = None, num_images=8
    ):
        """Initialize with algorithm setup."""
        if config is None:
            config = DEFAULT_CONFIG
        self.config = _validate_config(config)

        self.model_name = model_name
        self.setup = dict(device=device, dtype=torch.float32)
        self.mean_std = mean_std
        self.num_images = num_images

        self.device = device
        # self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup).to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def reconstruct(
            self, target_gradient: List[torch.Tensor],
            labels: torch.Tensor, img_shape=(3, 32, 32),
            dryrun=False, eval=True, tol=None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.server_model.eval()
            # self.client_model.eval()

        stats = {}
        x = self._init_images(img_shape).to(self.device)

        scores = torch.zeros(self.config['restarts']).to(self.device)

        assert labels.shape[0] == self.num_images

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], target_gradient, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, target_gradient, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interrupted.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, target_gradient, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time() - start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape: Tuple[int]) -> Union[torch.Tensor, float]:
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial: torch.Tensor, target_gradient: List[torch.Tensor], labels: torch.Tensor,
                   dryrun=False):
        x_trial.requires_grad = True
        if self.config['optim'] == 'adam':
            optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
        elif self.config['optim'] == 'sgd':  # actually gd
            optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
        elif self.config['optim'] == 'LBFGS':
            optimizer = torch.optim.LBFGS([x_trial])
        else:
            raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_iterations // 2.667, max_iterations // 1.6,
                    max_iterations // 1.142
                ],
                gamma=0.1
            )  # 3/8 5/8 7/8

        iteration = 0
        try:
            min_loss=999999999
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, target_gradient, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        if min_loss > rec_loss:

                            print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(
            self, optimizer: torch.optim.Optimizer,
            x_trial: torch.Tensor,
            target_gradient: List[torch.Tensor],
            label: torch.Tensor
    ) -> Callable[[], torch.Tensor]:
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            # self.client_model.zero_grad()
            self.server_model.zero_grad()
            loss = self.loss_fn(self.server_model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.server_model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs(
                [gradient], target_gradient,
                cost_fn=self.config['cost_fn'],
                indices=self.config['indices'],
                weights=self.config['weights']
            )

            if TOTAL_VARIATIONS[self.model_name] > 0:
                rec_loss += TOTAL_VARIATIONS[self.model_name] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(
            self, x_trial: torch.Tensor,
            target_gradient: List[torch.Tensor],
            label: torch.Tensor
    ) -> Union[torch.Tensor, float]:
        if self.config['scoring_choice'] == 'loss':
            self.server_model.zero_grad()
            # self.client_model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.server_model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.server_model.parameters(), create_graph=False)
            return reconstruction_costs(
                [gradient], target_gradient,
                cost_fn=self.config['cost_fn'],
                indices=self.config['indices'],
                weights=self.config['weights']
            )
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return self._average_trials(x, labels, input_data, stats)
        else:
            raise ValueError()

    def _average_trials(
            self, x: torch.Tensor,
            labels: torch.Tensor,
            target_gradient: List[torch.Tensor],
            stats: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)
        else:
            raise ValueError('Invalid scoring choice!')

        # self.client_model.zero_grad()
        self.server_model.zero_grad()
        loss = self.loss_fn(self.server_model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.server_model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs(
            [gradient], target_gradient,
            cost_fn=self.config['cost_fn'],
            indices=self.config['indices'],
            weights=self.config['weights']
        )
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


def reconstruction_costs(
        trial_gradients: List[Tuple[torch.Tensor]],
        target_gradient: Tuple[torch.Tensor],
        cost_fn='l2', indices='def', weights='equal'
) -> torch.Tensor:
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(target_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(target_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in target_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in target_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in target_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(target_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(target_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(target_gradient))[-50:]
    else:
        raise ValueError()

    ex = target_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(target_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(target_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(target_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = target_gradient[0].new_ones(len(target_gradient))

    total_costs = 0
    for trial_gradient in trial_gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - target_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - target_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - target_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * target_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += target_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   target_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs

    return total_costs / len(trial_gradients)


def privacy_score(x_trial: torch.Tensor, x: torch.Tensor):
    """cosine similarity of the ground truth image and reconstructed image"""
    x_trial = x_trial.flatten().unsqueeze(0)
    x = x.flatten().unsqueeze(0)
    return torch.nn.functional.cosine_similarity(x_trial, x).item()

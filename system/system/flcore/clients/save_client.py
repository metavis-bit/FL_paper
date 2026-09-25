# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import math
import os.path
import random
from pathlib import Path
import torch.nn.functional as F
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import torchvision
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import S_SCAFFOLDOptimizer
from flcore.attacks.adaptive import save_adaptive_observation
from flcore.servers.inversefed import consts
from flcore.servers.reconstructor import GradientReconstructor, privacy_score
from flcore.servers.inversefed.pytorch_ssim_master import pytorch_ssim
import torchvision.transforms as transforms
import torch.nn as nn
import copy
import math
random.seed(1)
total_layer = 62
min_decision_l = 6


class exp_client(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = S_SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = S_SCAFFOLDOptimizer(self.model.parameters(), lr=1)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        self.client_c = []
        self._client_c_initialized = False
        self.global_c = None
        self.global_model = None

        self.energy_cost_max = 80
        self.user_privacy_max = 0
        self.model_accuracy_max = 0
        self.decision_l = 6
        # self.decision_l = 38
        self.ratio = 1
        self.p_count = 0
        self.init_model_flag = True
        self._adaptive_first_batch = None

        # The paper initializes the private client control variate with the
        # local gradient at the initial model; it is never uploaded.
        self._initialize_client_c()

    def _initialize_client_c(self):
        if self._client_c_initialized:
            return

        loader = self.load_train_data()
        self.model.to(self.device)
        was_training = self.model.training
        buffer_state = {
            name: value.detach().clone() for name, value in self.model.named_buffers()
        }
        self.model.train()
        sums = [torch.zeros_like(param) for param in self.model.parameters()]
        batches = 0
        try:
            for x, y in loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.model.zero_grad(set_to_none=True)
                gradients = torch.autograd.grad(self.loss(self.model(x), y), self.model.parameters())
                for total, gradient in zip(sums, gradients):
                    total.add_(gradient.detach())
                batches += 1
            if batches == 0:
                raise RuntimeError(f"client {self.id} has no batch for cc initialization")
            self.client_c = [total / batches for total in sums]
        finally:
            with torch.no_grad():
                for name, value in self.model.named_buffers():
                    value.copy_(buffer_state[name])
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
        self._client_c_initialized = True

    def train(self):
        trainloader = self.load_train_data()
        self.num_batches = len(trainloader)
        capture_dir = getattr(self.args, "adaptive_capture_dir", None)
        capture_client = getattr(self.args, "adaptive_capture_client", None)
        capture = bool(capture_dir) and (
            capture_client is None or int(capture_client) == int(self.id)
        )
        self._adaptive_first_batch = None
        pre_model_state = (
            {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
            if capture else None
        )

        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs


        # self.decision_l = 62
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.loss(output, y)
                loss.backward()
                if capture and self._adaptive_first_batch is None:
                    self._adaptive_first_batch = {
                        "sc": [value.detach().clone() for value in self.global_c],
                        "images": x.detach().clone(),
                        "labels": y.detach().clone(),
                    }
                # if self.id == 0:
                #     target_gradient, _ = self.delta_yc()
                #     total_score = np.mean(self.reconstructor_process(target_gradient, trainloader))
                #     print("total_score:", total_score)
                self.optimizer.step(self.global_c, self.client_c, self.decision_l)



        self.update_yc(max_local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        if capture and self._adaptive_first_batch is not None:
            _, delta_cc = self.delta_yc()
            round_index = int(getattr(self, "_adaptive_round", 0))
            output_path = Path(capture_dir) / (
                f"round_{round_index:04d}_client_{int(self.id):04d}.pt"
            )
            save_adaptive_observation(
                output_path,
                self.model,
                self._adaptive_first_batch["sc"],
                delta_cc,
                self._adaptive_first_batch["images"],
                self._adaptive_first_batch["labels"],
                self.learning_rate,
                self.num_batches,
                max_local_epochs,
                round_index,
                int(self.id),
                self.decision_l,
                num_classes=self.num_classes,
                model_state_dict=pre_model_state,
            )

    def set_parameters(self, model, global_c):

        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = copy.deepcopy(global_c)
        self.global_model = model


    def update_yc(self, max_local_epochs):
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi)


    def delta_yc(self):
        max_local_epochs = self.local_epochs
        delta_y = []
        delta_c = []
        self.model.to(self.device)
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(-c + 1/self.num_batches/max_local_epochs / self.learning_rate * (x - yi))

        return delta_y, delta_c

    # def reconstructor_process(self, target_gradient, trainloader):
    #     # print(dataset)
    #     model_name = 'ResNet18'
    #     global_model = copy.deepcopy(self.model)
    #     device = self.device
    #     dm = torch.as_tensor(consts.cifar10_mean, device=device)[:, None, None]
    #     ds = torch.as_tensor(consts.cifar10_std, device=device)[:, None, None]
    #     total_score = []
    #
    #     for i, (x, y) in enumerate(trainloader):
    #         print("***********************************")
    #         if type(x) == type([]):
    #             x[0] = x[0].to(self.device)
    #         else:
    #             x = x.to(self.device)
    #         y = y.to(self.device)
    #
    #         x_ori, y_ori = copy.deepcopy(x), copy.deepcopy(y)
    #         loss_fn = nn.CrossEntropyLoss().to(device)
    #         # # x.to('device')
    #         target_loss = loss_fn(global_model(x), y)
    #         target_gradient = torch.autograd.grad(target_loss, global_model.parameters())
    #         step = 0
    #         for cc, sc, gradient in zip(self.client_c, self.global_c, target_gradient):
    #             if step >= total_layer - self.decision_l:
    #                 gradient.data = (gradient + sc - cc)
    #                 step += 1
    #             else:
    #                 gradient.data = gradient
    #                 step += 1
    #
    #         target_gradient = [grad.detach() for grad in target_gradient]
    #         reconstructor = GradientReconstructor(model_name, global_model, device, (dm, ds), num_images=x.shape[0])
    #         x_hat, stats = reconstructor.reconstruct(target_gradient, y, x.shape[1:])
    #         x_hat_index = x_hat.cpu()
    #         score = pytorch_ssim.ssim(x_ori, x_hat).to('cpu')
    #         total_score.append(score)
    #         self.imshow(torchvision.utils.make_grid(x_ori), 0.0)
    #         self.imshow(torchvision.utils.make_grid(x_hat_index), stats['opt'])
    #
    #     return total_score

    def reconstructor_process(self, target_gradient, trainloader):
        model_name = 'ResNet18'
        global_model = copy.deepcopy(self.model)
        device = self.device
        dm = torch.as_tensor(consts.cifar10_mean, device=device)[:, None, None]
        ds = torch.as_tensor(consts.cifar10_std, device=device)[:, None, None]

        # --- Setup for metrics collection and saving ---
        total_ssim, total_mse, total_psnr, total_recognition_rate, total_confidence = [], [], [], [], []

        # Create the directory for saving results if it doesn't exist
        path = "privacy/" + str(self.args.algorithm) + self.args.model_name + "_" + str(self.decision_l)
        if not os.path.exists(path):
            os.makedirs(path)

        # Define file path for the metrics report
        metrics_file_path = os.path.join(path, "reconstruction_metrics.txt")
        # --- End Setup ---

        for i, (x, y) in enumerate(trainloader):
            print("***********************************")
            # Limit processing to a few batches for efficiency during demonstration.
            # Remove this 'if' block to evaluate on the entire dataset.
            if i >= 2:
                break

            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            x_ori, y_ori = copy.deepcopy(x), copy.deepcopy(y)

            loss_fn = nn.CrossEntropyLoss().to(device)
            target_loss = loss_fn(global_model(x), y)
            target_gradient_tuple = torch.autograd.grad(target_loss, global_model.parameters())

            # Modify gradients based on SCAFFOLD logic
            step = 0
            modified_gradient = []
            for cc, sc, gradient in zip(self.client_c, self.global_c, target_gradient_tuple):
                grad_clone = gradient.clone()
                if step >= total_layer - self.decision_l:
                    grad_clone.data += sc.data - cc.data
                modified_gradient.append(grad_clone)
                step += 1

            target_gradient_detached = [grad.detach() for grad in modified_gradient]

            reconstructor = GradientReconstructor(model_name, global_model, device, (dm, ds), num_images=x.shape[0])
            x_hat, stats = reconstructor.reconstruct(target_gradient_detached, y, x.shape[1:])

            x_hat = x_hat.to(device)

            # --- METRICS CALCULATION ---

            # 1. Mean Squared Error (MSE)
            mse_val = F.mse_loss(x_hat, x_ori).item()
            total_mse.append(mse_val)

            # 2. Peak Signal-to-Noise Ratio (PSNR)
            # PSNR is calculated on images with pixel values typically in [0, 1].
            # We scale the tensors from [-1, 1] to [0, 1].
            x_ori_norm = (x_ori + 1) / 2
            x_hat_norm = (x_hat + 1) / 2
            mse_for_psnr = F.mse_loss(x_hat_norm, x_hat_norm)
            if mse_for_psnr.item() == 0:
                psnr_val = float('inf')
            else:
                psnr_val = 20 * math.log10(1.0 / math.sqrt(mse_for_psnr.item()))
            total_psnr.append(psnr_val)

            # 3. Structural Similarity (SSIM)
            ssim_val = pytorch_ssim.ssim(x_ori, x_hat).item()
            total_ssim.append(ssim_val)

            # 4. Recognition Rate & 5. Downstream Classifier Confidence
            global_model.eval()
            with torch.no_grad():
                output = global_model(x_hat)
                softmax_output = F.softmax(output, dim=1)

                confidence_vals, predicted_labels = torch.max(softmax_output, 1)
                total_confidence.append(confidence_vals.mean().item())

                correct_predictions = (predicted_labels == y_ori).sum().item()
                recognition_rate = correct_predictions / y_ori.size(0)
                total_recognition_rate.append(recognition_rate)

            # --- END METRICS CALCULATION ---

            # Save image samples (as in original code)
            self.imshow(torchvision.utils.make_grid(x_ori.cpu()), 0.0)
            self.imshow(torchvision.utils.make_grid(x_hat.cpu()), ssim_val)

        # --- SAVE METRICS TO FILE ---
        avg_mse = np.mean(total_mse)
        avg_psnr = np.mean(total_psnr)
        avg_ssim = np.mean(total_ssim)
        avg_recognition_rate = np.mean(total_recognition_rate)
        avg_confidence = np.mean(total_confidence)

        # Append results to the text file
        with open(metrics_file_path, "a") as f:
            f.write(f"--- Results from reconstruction run ---\n")
            f.write(f"Decision Layer Setting: {self.decision_l}\n")
            f.write(f"Average Mean Squared Error (MSE): {avg_mse:.4f}\n")
            f.write(f"Average Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB\n")
            f.write(f"Average Structural Similarity (SSIM): {avg_ssim:.4f}\n")
            f.write(f"Average Recognition Rate on Reconstructed Data: {avg_recognition_rate:.4f}\n")
            f.write(f"Average Downstream Classifier Confidence: {avg_confidence:.4f}\n")
            f.write("-" * 35 + "\n\n")

        print(f"Reconstruction metrics have been saved to {metrics_file_path}")

        # Return SSIM scores to maintain original function's behavior
        return total_ssim

    def imshow(self, img, score):
        img=img.to('cpu')
        img = img / 2 + 0.5  # 反归一化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换维度以适应matplotlib
        # plt.show()
        path = "privacy/" + str(self.args.algorithm)+self.args.model_name + "_" + str(self.decision_l)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + "{}_img_{}_score_{:.2f}.png".format(self.args.algorithm,self.p_count, score), format='png', dpi=600)
        self.p_count += 1

# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

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

import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
## privacy
from flcore.servers.inversefed import consts
import torchvision
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from flcore.servers.reconstructor import GradientReconstructor, privacy_score
from flcore.servers.inversefed.pytorch_ssim_master import pytorch_ssim
import torch.nn.functional as F
import math


class clientdpavg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.p_count = 0

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to('cpu')
        self.model.to(self.device)
        self.model.train()


        start_time = time.time()
        # # == == == == == == == == == == = 攻击模拟开始(修改后) == == == == == == == == == == =
        # if self.id == 0 and self.args.safe_flag == True:
        #     # 1. 为了不影响正常的训练流程，我们使用迭代器单独获取一个批次用于攻击
        #     try:
        #         attack_loader_iterator = iter(trainloader)
        #         x_attack, y_attack = next(attack_loader_iterator)
        #         x_attack, y_attack = x_attack.to(self.device), y_attack.to(self.device)
        #     except StopIteration:
        #         print("数据加载器为空，跳过重构攻击。")
        #         # 如果数据加载器为空，则直接进行后续的正常训练
        #         pass
        #     else:
        #         # 2. 深度拷贝一份“训练前”的模型状态
        #         print("正在模拟梯度反演攻击...")
        #         initial_model = copy.deepcopy(self.model)
        #         initial_model.eval()  # 切换到评估模式以确保行为一致
        #
        #         # 3. 在“训练前”的模型上计算目标梯度
        #         loss_fn = nn.CrossEntropyLoss().to(self.device)
        #         target_loss = loss_fn(initial_model(x_attack), y_attack)
        #         target_gradient = torch.autograd.grad(target_loss, initial_model.parameters())
        #         target_gradient = [grad.detach() + torch.randn_like(grad) * self.args.noise_multiplier for grad in target_gradient]
        #
        #         # 4. 调用重构函数，传入正确的信息
        #         # 注意：这里传入的是 initial_model, target_gradient, 以及原始数据
        #         mean_ssim_score = np.mean(self.reconstructor_process(
        #             initial_model,
        #             target_gradient,
        #             x_attack,
        #             y_attack
        #         ))
        #         print(f"攻击模拟完成。平均SSIM分数: {mean_ssim_score:.4f}")
        #     print("Done!")
        # # ===================== 攻击模拟结束 =====================
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        initial_model = copy.deepcopy(self.model)
        initial_model.eval()  # 切换到评估模式以确保行为一致
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                gradients = torch.autograd.grad(loss, initial_model.parameters())
                target_gradients = [grad.detach().clone() for grad in gradients]
                mean_ssim_score = np.mean(self.reconstructor_process(
                    initial_model,
                    target_gradients,
                    x,
                    y
                ))

                self.optimizer.step()



        # 添加高斯噪声
        if self.args.noise_multiplier > 0:
            for param in self.model.parameters():
                noise = torch.randn_like(param) * self.args.noise_multiplier
                param.data += noise * self.learning_rate

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def reconstructor_process(self, server_model, target_gradient, x_ori, y_ori):
        model_name = 'ResNet18'
        global_model = copy.deepcopy(self.model)
        device = self.device
        dm = torch.as_tensor(consts.cifar10_mean, device=device)[:, None, None]
        ds = torch.as_tensor(consts.cifar10_std, device=device)[:, None, None]

        # --- Setup for metrics collection and saving ---
        total_ssim, total_mse, total_psnr, total_recognition_rate, total_confidence = [], [], [], [], []

        # Create the directory for saving results if it doesn't exist
        path = "privacy/" + self.args.algorithm + "/" + self.args.model_name + "_" + str(self.args.noise_multiplier)
        if not os.path.exists(path):
            os.makedirs(path)

        # Define file path for the metrics report
        metrics_file_path = os.path.join(path, "reconstruction_metrics.txt")
        # --- End Setup ---

        # for i, (x, y) in enumerate(trainloader):
        #     print("***********************************")
        #     # Limit processing to a few batches for efficiency during demonstration.
        #     # Remove this 'if' block to evaluate on the entire dataset.
        #     if i >= 2:
        #         break
        #
        #     if type(x) == type([]):
        #         x[0] = x[0].to(self.device)
        #     else:
        #         x = x.to(self.device)
        #     y = y.to(self.device)
        #
        #     x_ori, y_ori = copy.deepcopy(x), copy.deepcopy(y)
        #     loss_fn = nn.CrossEntropyLoss().to(device)
        #     target_loss = loss_fn(global_model(x), y)
        #     target_gradient = torch.autograd.grad(target_loss, global_model.parameters())
        #
        #     for param, gradient in zip(global_model.parameters(), target_gradient):
        #         noise = torch.randn_like(param) * self.args.noise_multiplier
        #         gradient += noise

        # target_gradient = [grad.detach() for grad in target_gradient]
        reconstructor = GradientReconstructor(model_name, global_model, device, (dm, ds), num_images=x_ori.shape[0])
        x_hat, stats = reconstructor.reconstruct(target_gradient, y_ori, x_ori.shape[1:])

        x_hat = x_hat.to(device)

        # --- METRICS CALCULATION ---

        # 1. Mean Squared Error (MSE)
        mse_val = F.mse_loss(x_hat, x_ori).item()
        total_mse.append(mse_val)

        # 2. Peak Signal-to-Noise Ratio (PSNR)
        # Scale tensors from [-1, 1] to [0, 1] for PSNR calculation.
        x_ori_norm = (x_ori + 1) / 2
        x_hat_norm = (x_hat + 1) / 2
        mse_for_psnr = F.mse_loss(x_ori_norm, x_hat_norm)
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

        # Update image saving to use the calculated SSIM score
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
            f.write(f"--- Results from {self.args.algorithm} reconstruction run ---\n")
            f.write(f"Noise Multiplier: {self.args.noise_multiplier}\n")
            f.write(f"Average Mean Squared Error (MSE): {avg_mse:.4f}\n")
            f.write(f"Average Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB\n")
            f.write(f"Average Structural Similarity (SSIM): {avg_ssim:.4f}\n")
            f.write(f"Average Recognition Rate on Reconstructed Data: {avg_recognition_rate:.4f}\n")
            f.write(f"Average Downstream Classifier Confidence: {avg_confidence:.4f}\n")
            f.write("-" * 35 + "\n\n")

        print(f"Reconstruction metrics have been saved to {metrics_file_path}")

        # Return SSIM scores to maintain original behavior for the train function
        return total_ssim

    def imshow(self, img, score):
        img = img.to('cpu')
        img = img / 2 + 0.5  # 反归一化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换维度以适应matplotlib
        # plt.show()
        path = "privacy/" + self.args.algorithm + "/" + self.args.model_name + "_" + str(self.args.noise_multiplier)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + "{}_img_{}_score_{:.2f}.png".format(self.args.algorithm,self.p_count, score), format='png', dpi=600)
        self.p_count += 1

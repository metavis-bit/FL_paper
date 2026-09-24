# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import math
import random

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
from flcore.servers.inversefed import consts
from flcore.servers.reconstructor import GradientReconstructor, privacy_score
from flcore.servers.inversefed.pytorch_ssim_master import pytorch_ssim
import torchvision.transforms as transforms
import torch.nn as nn
import copy
total_sample = 10
random.seed(1)
total_layer = 58
min_decision_l = 40


class clientS_SCAFFOLD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = S_SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = S_SCAFFOLDOptimizer(self.model.parameters(), lr=1)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

        self.energy_cost_max = 80
        self.user_privacy_max = 0
        self.model_accuracy_max = 0
        # self.decision_l = random.randint(min_decision_l, total_layer)0
        self.decision_l = random.randint(min_decision_l, total_layer)
        # self.decision_l = 62
        self.ratio = 1
        self.p_count = 0
        self.init_model_flag = True

    def decision_l_make(self, client_decision_list, average_decision_l):
        # 决策前的cost

        total_model_size = 0
        model_size = 0
        count = 0

        # last_train_cost = self.caculate_cost(ratio=self.decision_l/38)
        last_train_cost = self.caculate_cost()
        # # print("last_train_cost:", last_train_cost)
        last_decision = self.decision_l
        # 如果其他用户的决策足够达到目标精度,则用户可以尽可能多的添加方差控制保护自己的隐私,反之则要减少方差控制达到目标精度

        if np.average(client_decision_list) <= int(average_decision_l):
            self.decision_l += max(min(average_decision_l - np.average(client_decision_list), total_layer), 1)
            if last_train_cost < self.caculate_cost():
                self.decision_l = last_decision
        else:
            self.decision_l -= max(min(average_decision_l - np.average(client_decision_list), total_layer), 1)
            if last_train_cost < self.caculate_cost():
                self.decision_l = last_decision

        self.decision_l = int(max(min(self.decision_l, total_layer), min_decision_l))
        # # 决策后的cost
        #
        # self.decision_l = min_decision_l # for FedPVR

        for name, param in self.model.named_parameters():
            count += 1
            total_model_size += param.numel()
            if count > total_layer - 6 or count < total_layer - 6:
                model_size += param.numel()
        self.ratio = model_size / total_model_size
        # print("ratio = ", self.ratio)

        # predict_cost = (round_index - g_round) * 3
        # predict_cost = float(self.slove_next_round(self.num_batches, self.global_acc_history[-1]) * 3)
        predict_cost = 0
        # print(predict_cost)
        self.modelAcc_history.append(self.model_accuracy(ratio=self.ratio))
        self.privacy_history.append(self.user_privacy(ratio=self.ratio))
        # print("privacy {}:".format(self.id),self.privacy_history[-1])
        self.energy_cost_history.append(self.energy_cost(ratio=self.ratio))
        self.trained_cost.append(self.caculate_cost(ratio=self.ratio))
        self.train_cost.append(sum(self.trained_cost))
        self.predict_cost_history.append(predict_cost)

    def train(self,client_decision_list, average_decision_l):
        trainloader = self.load_train_data()
        self.num_batches = len(trainloader)

        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs

        if self.algorithm == "FedPVR":
            self.decision_l = 6

        self.decision_l_make(client_decision_list, average_decision_l)

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
                self.optimizer.step(self.global_c, self.client_c, self.decision_l)
                # self.optimizer.step()

            # count = 0
            # for param, sc, cc in zip(self.model.parameters(), self.global_c, self.client_c):
            #     if count < self.args.total_layer - self.decision_l:
            #         continue
            #     else:
            #         param.data = param.data - self.learning_rate * (sc - cc)
            #     count += 1

        self.update_yc(max_local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model, global_c):

        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = copy.deepcopy(global_c)
        self.global_model = model


    def update_yc(self, max_local_epochs):
        self.num_batches = max(1, self.num_batches)
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
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += (y.shape[0] - total_sample)

                # y_prob.append(output.detach().cpu().numpy())
                # nc = self.num_classes
                # if self.num_classes == 2:
                #     nc += 1
                # lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                # if self.num_classes == 2:
                #     lb = lb[:, :2]
                # y_true.append(lb)

        # y_prob = np.concatenate(y_prob, axis=0)
        # y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = None

        # self.global_acc_history.append(test_acc / test_num)

        self.model.to('cpu')
        return test_acc, test_num, auc
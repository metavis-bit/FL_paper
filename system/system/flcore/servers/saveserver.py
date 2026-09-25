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
import random
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from flcore.clients.save_client import exp_client
from flcore.servers.serverbase import Server
from flcore.servers.reconstructor import GradientReconstructor, privacy_score
from flcore.servers.inversefed import consts
import torch.nn as nn
from flcore.servers.inversefed.pytorch_ssim_master import pytorch_ssim
import numpy as np
from threading import Thread
import torchvision.transforms as transforms
from sklearn.preprocessing import label_binarize
from flcore.privacy_cost import privacy_cost
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader

ratio = 0.1
total_layer = 58
min_decision = 6

class exp(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(exp_client)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # self.load_model()
        self.server_learning_rate = args.server_learning_rate
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))

        self.round_index = 5
        self.total_cost = []
        self.average_decision_l = 62
        self.slope = 0
        self.total_train_cost = []
        self.p_count = 0
        self.rs_test_acc=[0]

        self.old_model = self.global_model

    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()

            for c in self.selected_clients:
                c.model.to(c.device)

            client_decision_list = []
            for client in self.selected_clients:
                client_decision_list.append(client.decision_l)

            if self.args.algorithm == "Adap-CTA":
                total_train_cost = self.make_clients_decision(client_decision_list)
                self.total_train_cost.append(sum(total_train_cost))

            for client in self.selected_clients:
                client._adaptive_round = i
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.send_models()
                self.evaluate()
            #
            # if i > 100:
            #     self.shuffle_layers()

            # self.average_decision_l = max(math.sqrt(self.average_decision_l), 2)  ## resnet10 + Cifar10
            self.average_decision_l = max(4, self.average_decision_l - 1)  ## resnet10 + Cifar10
            # self.average_decision_l = min(max(self.average_decision_l + 1, 10), 62)  ## resnet10 + Cifar10
            for c in self.selected_clients:
                c.model.to('cpu')
            self.Budget.append(time.time() - s_t)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientSCAFFOLD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_models(self, train_flag=False):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model, self.global_c)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        # assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        active_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        # self.delta_ys = []
        # self.delta_cs = []
        for client in self.selected_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.delta_ys.append(client.delta_y)
                # self.delta_cs.append(client.delta_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        # original version
        # for dy, dc in zip(self.delta_ys, self.delta_cs):
        #     for server_param, client_param in zip(self.global_model.parameters(), dy):
        #         server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
        #     for server_param, client_param in zip(self.global_c, dc):
        #         server_param.data += client_param.data.clone() / self.num_clients

        # save GPU memory
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for client in self.selected_clients:
            dy, dc = client.delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.num_clients

        self.global_model = global_model
        self.global_c = global_c

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.global_c)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def make_clients_decision(self, last_decision_list):
        # 利用潜在势博弈的方法推导出self.selected_clients的每个client的决策;
        # 每个client的决策是一个标量，范围属于[min_decision, total_layer]
        # 总成本函数包括三部分：隐私成本、能耗成本和模型性能成本
        # 其中： ratio := (1 - self.average_decision_l / total_layer) * 100
        # Privacy cost is parameter-free: exp(-correction ratio).
        # 能耗成本为：energy_cost = (0.82 * clients.num_batches / (60000/total_client/ batch_size) + 0.09 + ratio * 0.09) 其中num_batches为客户端本地的batch数, ratio为客户端通过决策后添加矫正项的比例
        # 模型性能成本为：model_cost = max(0, (client.target_acc - self.rs_test_acc[-1])

        new_decision_list = [0] * len(self.selected_clients)
        array1 = np.array(new_decision_list)
        array2 = np.array(last_decision_list)
        distance = np.linalg.norm(array1 - array2)
        round_total_cost = [0] * len(self.selected_clients)

        while distance > 0.05:
            round_total_cost = [0] * len(self.selected_clients)
            for count, client in enumerate(self.selected_clients):
                if self.calculate_cost(last_decision_list[count]-1, client, new_decision_list, count)[0] < self.calculate_cost(client.decision_l, client, new_decision_list, count)[0]:
                    client.decision_l = max(min(last_decision_list[count]-1, total_layer), min_decision)
                elif self.calculate_cost(last_decision_list[count] + 1, client, new_decision_list, count)[0] <= self.calculate_cost(client.decision_l, client, new_decision_list, count)[0]:
                    client.decision_l = max(min(last_decision_list[count]+1, total_layer), min_decision)
                round_total_cost[count], privacy_cost, energy_cost, model_cost = self.calculate_cost(client.decision_l, client, new_decision_list, count)
                new_decision_list[count] = client.decision_l
                last_decision_list[count] = new_decision_list[count]

            array1 = np.array(new_decision_list)
            array2 = np.array(last_decision_list)
            distance = np.linalg.norm(array1 - array2)

        return round_total_cost

    def calculate_cost(self, decision_l, client, new_decision_list, count):
        # 计算ratio：

        total_model_size = model_size = 0

        list = new_decision_list[:]
        list[count] = decision_l
        step = 0
        average_decision_l = int(sum(list) / len(list))
        for param in self.global_model.parameters():
            step += 1
            total_model_size += param.numel()
            if step > total_layer - average_decision_l:
                model_size += param.numel()
        ratio = model_size / total_model_size

        # 计算隐私成本
        privacy_value = privacy_cost(ratio)
        # 计算能耗成本
        energy_cost = (0.82 * client.num_batches / (60000 / self.args.num_clients / self.args.batch_size) + 0.09 + ratio * 0.09)
        # 计算模型性能成本
        model_cost = max(0, (client.target_acc - self.rs_test_acc[-1]))
        # 计算总成本
        total_cost = privacy_value + energy_cost + model_cost
        return total_cost, privacy_value, energy_cost, model_cost

    def reconstructor_process(self, model, dataset):
        model_name = 'ResNet10'
        global_model = model
        device = self.device
        dm = torch.as_tensor(consts.cifar10_mean, device=device)[:, None, None]
        ds = torch.as_tensor(consts.cifar10_std, device=device)[:, None, None]
        total_score = []
        for count, (x, y) in enumerate(dataset):
            x_index, y_index = x, y
            x, y = x.to(device), y.to(device)
            if count == 4:
                break

            global_model.eval()

            self.imshow(torchvision.utils.make_grid(x_index))
            loss_fn = nn.CrossEntropyLoss().to(device)
            # x.to('device')
            target_loss = loss_fn(global_model(x), y)
            target_gradient = torch.autograd.grad(target_loss, global_model.parameters())
            target_gradient = [grad.detach() for grad in target_gradient]

            reconstructor = GradientReconstructor(model_name, global_model, device, (dm, ds))
            x_hat, stats = reconstructor.reconstruct(target_gradient, y, x.shape[1:])
            x_hat_index = x_hat.cpu()
            # x_hat_index.data.cpu().numpy()

            self.imshow(torchvision.utils.make_grid(x_hat_index))

            score = pytorch_ssim.ssim(x, x_hat).to('cpu')
            total_score.append(score)
        return total_score

    def imshow(self, img):
        img = img / 2 + 0.5  # 反归一化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换维度以适应matplotlib
        plt.show()
        plt.savefig("img_{}_.png".format(self.p_count), format='png', dpi=300)
        self.p_count += 1

    def evaluate(self, acc=None, loss=None):
        total_samples = 0
        total_correct = 0
        for client in self.clients:
            model = copy.deepcopy(self.global_model)
            model.to(self.device)
            testloaderfull = client.load_test_data()

            test_acc = 0
            test_num = 0
            y_prob = []
            y_true = []

            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

            total_samples += test_num
            total_correct += test_acc

        test_acc = total_correct / total_samples * 1.0
        train_loss = 0

        if acc == None:
            self.rs_test_acc.append(test_acc)
            NIID = 'NIID-' + str(self.args.NIID)
            self.save_instant_result(self.rs_test_acc, NIID)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Test Accurancy: {:.4f}".format(test_acc))

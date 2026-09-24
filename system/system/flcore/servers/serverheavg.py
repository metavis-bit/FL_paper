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

import time
import copy
import torch
from flcore.clients.clientheavg import clientHEAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
from . import paillier

sigma = 0.01
dp = 1
noise = []

def set_noise(model, args):
    for param in model.parameters():
        noise_index = torch.cuda.FloatTensor(param.shape).normal_(0, sigma).to(args.device)
        noise.append(noise_index)


def add_noise(parameters, dp, dev):
    noise = None
    # 不加噪声
    if dp == 0:
        return parameters
    # 拉普拉斯噪声
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape), dtype=torch.long).to(dev)
    # 高斯噪声
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)

    return parameters.add_(noise)


def encrypt_vector(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters


# list解密
def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters


class HEFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientHEAVG)
        # self.load_model()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        set_noise(self.global_model, self.args)


    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\n", self.args.algorithm)
                print("\nEvaluate global model")
                self.evaluate()

            for c in self.selected_clients:
                c.model.to(self.device)

            for client in self.selected_clients:
                _ = client.train()
            # for c in self.selected_clients:
            # clients_dict_collect = []
            # # global_model_dict = self.global_model.state_dict()
            # sum_parameters = None
            # public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
            #
            # for client in self.selected_clients:
            #     local_parameters = client.train()
            #     clients_dict_collect.append(local_parameters)
            #     print("client id:", client.id)
            #
            #
            #
            #     if sum_parameters is None:
            #         sum_parameters = {}
            #         parameters_shape = {}
            #         for key, var in local_parameters.items():
            #             sum_parameters[key] = var.clone()
            #             parameters_shape[key] = var.shape
            #             sum_parameters[key] = add_noise(sum_parameters[key], dp, self.args.device)
            #             print(1)
            #             sum_parameters[key] = encrypt_vector(public_key, sum_parameters[key])
            #             print(2)
            #
            #
            #     else:
            #         for key in sum_parameters:
            #             sum_parameters[key] = np.add(sum_parameters[key], encrypt_vector(public_key, add_noise(
            #                 local_parameters[key], dp, self.args.device)))
            #
            #     global_model_dict = sum_parameters

            # for var in global_parameters:
            #     sum_parameters[var] = decrypt_vector(private_key, sum_parameters[var])
            #     sum_parameters[var] = torch.reshape(torch.Tensor(sum_parameters[var]), parameters_shape[var])
            #     global_parameters[var] = (sum_parameters[var].to(dev) / num_in_comm)

            # self.global_model.load_state_dict(global_parameters)
            # sum_parameters = None
            self.receive_models()
            #
            self.aggregate_parameters()

            for c in self.selected_clients:
                c.model.to('cpu')

            self.Budget.append(time.time() - s_t)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])

        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for client in self.selected_clients:
            for s_param, c_param, noise_param in zip(self.global_model.parameters(), client.model.parameters(), noise):
                s_param.data = s_param.data.clone() + (c_param.data.clone() + noise_param) / len(self.selected_clients)

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

    def send_models(self):
        assert (len(self.clients) > 0)

        for param, noise_param in zip(self.global_model.parameters(), noise):
            param.data -= noise_param

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)









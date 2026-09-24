import copy

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client



class clientFedMut(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # # differential privacy
        # if self.privacy:
        #     check_dp(self.model)
        #     initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_train_parameters(self, model_dict):
        # w_locals_new = []
        # ctrl_cmd_list = []
        # ctrl_rate = self.args.mut_acc_rate * (1.0 - min(self.iter * 1.0 / self.args.mut_bound, 1.0))
        # # print(ctrl_rate)
        #
        # w_glob = copy.deepcopy(global_model.state_dict())
        #
        # ctrl_list = []
        # for k in w_glob.keys():
        #     ctrl = random.random()
        #     if ctrl > 0.5:
        #         ctrl_list.append(1.0)
        #     else:
        #         ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
        #     random.shuffle(ctrl_list)
        # w_sub = copy.deepcopy(self.model.state_dict())
        # step = 0
        # for k in w_sub.keys():
        #     w_sub[k] = w_glob[k] + (w_glob[k] - w_sub[k]) * ctrl_list[step] * self.args.MutAlpha
        #     step+=1

        self.model.load_state_dict(model_dict)


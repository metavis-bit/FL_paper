import copy
import time
import torch
from flcore.clients.clientFedMut import clientFedMut
from flcore.servers.serverbase import Server
from threading import Thread
import random


class FedMut(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedMut)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        local_acc = []
        w_local_new = []
        for _ in range(int(self.num_clients * self.join_ratio)):
            w_local_new.append(copy.deepcopy(self.global_model.state_dict()))

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            old_dict = copy.deepcopy(self.global_model.state_dict())

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\n", self.args.algorithm)
                print("\nEvaluate global model")
                self.evaluate()

            for j, client in enumerate(self.selected_clients):
                # self.send_train_models(w_local_new[j])
                client.set_train_parameters(w_local_new[j])
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            # if i%self.eval_gap == 0:
            #     print("\nEvaluate local model")
            #     self.evaluate(acc=local_acc)
            self.receive_models()
            w_glob = self.aggregate_parameters()
            w_local_new = self.mutModel(old_dict, w_glob, i)

            for client in self.selected_clients:
                client.model.cpu()

            self.Budget.append(time.time() - s_t)
            # print('-'*25, 'time cost', '-'*25, self.Budget[-1])


        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        # print("\nBest local accuracy.")
        # print(max(local_acc))
        self.save_results()
        # self.save_global_model()

    def mutModel(self, old_dict, w_glob, train_round):

        step = 1.0
        m = len(self.selected_clients)
        w_locals_new = []
        ctrl_cmd_list = []
        ctrl_rate = self.args.mut_acc_rate * (step - min(train_round * 1.0 / self.args.mut_bound, step))
        # print(ctrl_rate)
        for k in w_glob.keys():
            ctrl_list = []
            for i in range(0, int(m / 2)):
                ctrl = random.random()
                if ctrl > 0.5:
                    ctrl_list.append(step)
                    ctrl_list.append(step * (-step + ctrl_rate))
                else:
                    ctrl_list.append(step)
                    ctrl_list.append(step * (-step + ctrl_rate))
            random.shuffle(ctrl_list)
            ctrl_cmd_list.append(ctrl_list)
        cnt = 0
        for j in range(m):
            w_sub = copy.deepcopy(w_glob)
            if not (cnt == m - 1 and m % 2 == 1):
                ind = 0
                for k in w_sub.keys():
                    w_sub[k] = w_sub[k] + (w_glob[k] - old_dict[k]) * ctrl_cmd_list[ind][j] * self.args.MutAlpha
                    ind += 1
            cnt += 1
            w_locals_new.append(w_sub)

        return w_locals_new

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()
        #
        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     self.add_parameters(w, client_model)

        global_model_dict = copy.deepcopy(self.global_model.state_dict())
        client_step = 0
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_model_dict = copy.deepcopy(client_model.state_dict())
            if client_step == 0:
                for key in global_model_dict.keys():
                    global_model_dict[key] = client_model_dict[key] * (1 / len(self.selected_clients))
            else:
                for key in global_model_dict.keys():
                    global_model_dict[key] += client_model_dict[key] * (1 / len(self.selected_clients))
            client_step += 1

        self.global_model.load_state_dict(global_model_dict)
        return global_model_dict

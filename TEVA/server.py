import torch
import copy
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import CNNMnist, MLP
from threshold_paillier import *
import time
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, pair
# 初始化配对群
group = PairingGroup('SS512')
class Server():
    def __init__(self, args, w, dataset=None):
        self.args = args
        self.clients_loss = []
        self.img_size = dataset[0][0].shape
        if args.model_name == 'cnn':
            self.model = CNNMnist(args=args).to(args.device)
        else:
            len_in = 1
            for x in self.img_size:
                len_in *= x
            self.model = MLP(dim_in=len_in, dim_hidden=args.dim_hidden, dim_out=args.num_classes).to(args.device)
        self.model.load_state_dict(w)
        # DP hyperparameters
        self.C = self.args.C
        self.sigma = self.args.sigma

    def FedAvg(self, clients_update_w, v, a, clients_update_acc, clients_update_f, t, h, c):
        if self.args.mode == 'plain':
            print("clients_update_w[0]", clients_update_w[0])
            update_w_avg = copy.deepcopy(clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(clients_update_w)):
                    update_w_avg[k] += clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]   
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

        elif self.args.mode == 'DP':  # DP mechanism
            update_w_avg = copy.deepcopy(clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(clients_update_w)):
                    update_w_avg[k] += clients_update_w[i][k]
                # add gauss noise
                update_w_avg[k] += torch.normal(0, self.sigma**2 * self.C**2, update_w_avg[k].shape).to(self.args.device)
                update_w_avg[k] = torch.div(update_w_avg[k], len(clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

        elif self.args.mode == 'Threshold Paillier':
            if t % 3 == 0:
                update_w_avg = copy.deepcopy(clients_update_w[0])
                weig_w = copy.deepcopy(clients_update_w[0])
                delta_w = copy.deepcopy(clients_update_w[0])
                m_s = copy.deepcopy(clients_update_w[0])
                w_old = self.model.state_dict()
                update_v = v[0]
                #print('updat_v',update_v)
                update_a = a[0]
                update_c = c[0]
                acc_sum = 0
                f_sum = 0
                sum_acc = 0
                sum_f = 0
                acc_list = []
                f_list = []
                for k in update_w_avg.keys():
                    for i in range(len(update_w_avg[k])):
                        update_w_avg[k][i] = 0
                for i in range(len(clients_update_acc)):
                    acc_sum += clients_update_acc[i]
                    f_sum += clients_update_f[i]
                for i in range(len(clients_update_acc)):
                    acc_fed = clients_update_acc[i] / acc_sum
                    f_fed = clients_update_f[i] / f_sum
                    if acc_fed == 0:
                        acc = -math.log(acc_fed+self.args.nc, 2)
                    else:
                        acc = -math.log(acc_fed, 2)
                    sum_acc += acc
                    acc_list.append(acc)
                    if f_fed == 0:
                        f = -math.log(1 -f_fed+self.args.nc, 2)
                    else:
                        f = -math.log(1 - f_fed, 2)
                    sum_f += f
                    f_list.append(f)
                for i in range(len(acc_list)):
                    acc_inf = acc_list[i] / sum_acc
                    f_inf = f_list[i] / sum_f
                    weight_inf = self.args.alpha * acc_inf + self.args.beta * f_inf
                    for k in weig_w.keys():
                        for j in range(len(weig_w[k])):
                            v[i][k][j] = c[i][k][j] * v[i][k][j]
                for i in range(1, len(acc_list)):
                    for k in weig_w.keys():
                        for j in range(len(weig_w[k])):
                            update_w_avg[k][j] += clients_update_w[i][k][j]
                            update_v[k][j] *= v[i][k][j]

                for k in weig_w.keys():
                    for j in range(len(weig_w[k])):
                        update_v[k][j] = update_v[k][j] * update_c[k][j]
                #print('update_v', update_v)

                for k in update_w_avg.keys():
                    w_old_list = w_old[k].view(-1).cpu().tolist()
                    if t == 0:
                        m_s_list = self.m_s[k].view(-1).cpu().tolist()
                    else:
                        m_s_list = self.m_s[k]
                    for j in range(len(update_w_avg[k])):
                        delta_w[k][j] = update_w_avg[k][j] - w_old_list[j]
                        m_s_list[j] = self.args.server_grammar * m_s_list[j] + self.args.lr * delta_w[k][j]  # Update the momentum
                        update_w_avg[k][j] -= m_s_list[j]
                    self.m_s[k] = m_s_list
                    #origin_shape = list(self.model.state_dict()[k].size())
                    #m_s[k] = torch.FloatTensor(self.m_s[k]).to(self.args.device).view(*origin_shape)
                for k in weig_w.keys():
                    for j in range(len(weig_w[k])):
                        update_v[k][j] = update_v[k][j]/pow(pow(h,group.random(ZR, self.args.server_grammar)),group.random(ZR, self.m_s[k][j]))

                return update_w_avg, sum(self.clients_loss) / len(self.clients_loss), update_v, update_a

            else:
                update_w_avg = copy.deepcopy(clients_update_w[0])
                update_v = v[0]
                update_a = a[0]
                for k in update_w_avg.keys():
                    client_num = len(clients_update_w)
                    #print('client_num', client_num)
                    for i in range(1, client_num):  # client-wise sum
                        for j in range(len(update_w_avg[k])):  # element-wise sum
                            update_w_avg[k][j] += clients_update_w[i][k][j]
                            update_v[k][j] *= v[i][k][j]
                            update_a[k][j] *= a[i][k][j]
                for k in update_w_avg.keys():
                    client_num = len(clients_update_w)
                    for j in range(len(update_w_avg[k])):# element-wise avg
                        update_v[k][j] = pow(update_v[k][j], group.init(ZR, 1/client_num))
                        #update_v /= pow(h, -self.args.num_users)
                        update_w_avg[k][j] /= client_num


                return update_w_avg, sum(self.clients_loss) / len(self.clients_loss), update_v, update_a
    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss

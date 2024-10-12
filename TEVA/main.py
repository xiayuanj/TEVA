import time

from client import *
from server import *
import copy
from termcolor import colored
from models.Nets import CNNMnist, MLP, CNNCifar
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, pair
from sampling import *
from threshold_paillier import *
from pympler import asizeof
def Hash_res(number):
    # 将消息转换为群 G1 中的元素
    number_bytes = str(number).encode()
    # 计算哈希值
    H_0 = group.hash(number_bytes, G1)

    return H_0

def load_dataset():
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        print('args.iid:', args.iid)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users

def create_client_server():

    clients = []
    img_size = dataset_train[0][0].shape
    if args.model_name == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model_name == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=args.dim_hidden, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # divide training data, i.i.d.
    # init models with same parameters
    for idx in range(args.num_users):
        new_client = Client(args=args, dataset=dataset_train, idxs=dict_users[idx], w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()), dataset=dataset_train)

    return clients, server


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("Choose GPU or CPU:", args.device)

    print("Generate public and private keys...")
    public_key, private_key = generate_paillier_keypair()
    shares = private_key.split_key(args.num_users, args.threshold)

    print("load dataset...")
    dataset_train, dataset_test, dict_users = load_dataset()

    print("clients and server initialization...")
    clients, server = create_client_server()

    # statistics for plot
    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []

    # training
    print("start training...")
    print('Algorithm:', colored(args.mode, 'green'))
    print('Model:', colored(args.model_name, 'green'))

    for iter in range(args.epochs):
        #global_pub_key, global_priv_key = generate_paillier_keypair()
        n = public_key.n
        p = private_key.p
        group = PairingGroup('SS512')
        a = group.init(ZR, random.randint(1, p))
        sk = group.init(ZR, random.randint(1, n))
        g = group.random(G1)
        #print('g:', g)
        H_0_t = Hash_res(iter)
        h = g ** a
        epoch_start = time.time()
        server.clients_update_w, server.clients_loss, server.clients_V_t, server.clients_A_t, server.clients_acc, server.clients_f, server.clients_C_t = [], [], [], [], [], [], []
        #idxs_users = max(int(args.frac * args.num_users), 1)
        for idx in range(args.num_users):
            update_w, loss, V_t, A_t, acc, f, C_t = clients[idx].train(h, sk, g, H_0_t, iter, public_key)
            print('=====Client {:3d}====='.format(idx), acc)
            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)
            server.clients_V_t.append(V_t)
            server.clients_A_t.append(A_t)
            server.clients_acc.append(acc)
            server.clients_f.append(f)
            server.clients_C_t.append(C_t)
        fedavg_start = time.time()
        w_glob, loss_glob, V_glob, A_glob = server.FedAvg(server.clients_update_w, server.clients_V_t, server.clients_A_t, server.clients_acc, server.clients_f, iter, h, server.clients_C_t)
        #print('w_glob data size:', asizeof.asizeof(w_glob))
        fedavg_end = time.time()
        print('server computes time:', fedavg_end-fedavg_start)
        #print("w_glob:", w_glob)
        # update local weights
        for idx in range(args.num_users):
            clients[idx].update(w_glob, public_key, shares)
        if args.mode == 'Threshold Paillier':
            server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
        verify_w_glob = copy.deepcopy(clients[0].model.state_dict())
        for idx in range(args.num_users):
            #verify_start = time.time()
            clients[idx].verify(verify_w_glob, V_glob, g, H_0_t, A_glob, h, iter)
            verify_end = time.time()
            #print("Verify time:", verify_end-verify_start)
            print("==========Client{:3d} validation pass!==========".format(idx))

        #epoch_end = time.time()
        print(colored('=========================Epoch {:3d}========================='.format(iter), 'yellow'))
        #print('Training time:', epoch_end - epoch_start)
        # testing
        acc_train, loss_train = server.test(dataset_train)
        acc_test, loss_test = server.test(dataset_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print('Training average loss {:.3f}'.format(loss_glob))
        all_acc_train.append(acc_train)
        all_acc_test.append(acc_test)
        all_loss_glob.append(loss_glob)

    # plot learning curve
    if not args.no_plot:
        x = np.linspace(0, args.epochs - 1, args.epochs)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.suptitle('Learning curves of ' + args.mode)
        ax1.plot(x, all_acc_train)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train accuracy')
        ax2.plot(x, all_acc_test)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Testing accuracy')
        ax3.plot(x, all_loss_glob)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training average loss')
        plt.savefig('figs/' + args.mode + '_training_curve.png')
        plt.show()

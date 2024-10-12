#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # experiment arguments
    parser.add_argument('--mode', type=str, default='Threshold Paillier', help="plain, DP, or Threshold Paillier")
    parser.add_argument('--model_name', type=str, default='mlp', help='model name')
    parser.add_argument('--dim_hidden', type=int, default=2, help='number of dim_hidden')
    parser.add_argument('--rho', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--client_grammar', type=float, default=0.5, help='influence of the historical gradient: client')
    parser.add_argument('--server_grammar', type=float, default=0.5, help='influence of the historical gradient: server')
    parser.add_argument('--nc', type=float, default=0.000001, help='small constant')
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--alpha', type=float, default=0.8, help="the proportional effect of aggregated model: alpha")
    parser.add_argument('--beta', type=float, default=0.2, help="the proportional effect of aggregated model: beta")
    parser.add_argument('--key_size', type=int, default=256, help="the size of key")

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=2, help="number of users: K")
    parser.add_argument('--threshold', type=int, default=2, help="threshold size")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.015, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    
    # DP arguments
    parser.add_argument('--C', type=float, default=0.5, help="DP model clip parameter")
    parser.add_argument('--sigma', type=float, default=0.05, help="DP Gauss noise parameter")

    # other arguments
    parser.add_argument('--iid', action='store_true', default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--no-plot', action="store_true", default=False, help="plot learning curve")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    args = parser.parse_args()
    return args

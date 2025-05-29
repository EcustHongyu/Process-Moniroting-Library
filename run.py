import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from utils.print_args import print_args 


if __name__ == "__main__":
    fix_seed = 3407
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Your model name')

    # model definition
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='channel output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='number of multi-heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, 
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attention factor')
    parser.add_argument('--distil', action='store_false', default=True, 
                        help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--conf_level', type=float, default=0.95, 
                        help='confidence level of the kernel density estimation')
    
    # machine learning model definition
    parser.add_argument('--cum_ratio', type=float, default=0.75, help='cumulative covariance ratio of the principle components')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='kernel function of KPCA, options:[linear, rbf, poly, ...]')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels. If gamma is None, then it is set to 1/n_features.')
    parser.add_argument('--degree', type=int, default=3,
                        help='Degree for poly kernels. Ignored by other kernels')
    parser.add_argument('--time_lags', type=int, default=0, help='time lags of the dynamic feature matrix')
    parser.add_argument('--ratio_sfa', type=float, default=0.2, help='the ratio of slow feature in SFA')
    parser.add_argument('--input_dim', type=int, default=30, help='the dimension of input in sfa')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiment times')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers for dataloader')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of training input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='type of the learning rate adjustment')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--device', type=str, default='0, 1, 2, 3', help='device id of GPU')
    
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: {args.gpu}')
    else:
        args.device = torch.device('cpu')
        print('Using CPU')
    print('Arguments in experiment:')
    print_args(args)
    
    # Write your experiment here



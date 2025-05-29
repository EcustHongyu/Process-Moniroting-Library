import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from utils.print_args import print_args 
from exp.exp_fault_detection import Exp_fault_detection
from exp.exp_MLfault_detection import Exp_MLfault_detection
from exp.exp_KAEfault_detection import Exp_KAEfault_detection


if __name__ == "__main__":
    fix_seed = 3407
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Transformer')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='fault_detection',
                         help='task name, options:[fault_detection, fault_classification]')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                         help='model name, options:[Transformer, xxx]')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--is_training', type=int, required=True, default=1,
                         help='is training or testing')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', 
                        help='dataset name, options:[ETTh1, ETTh2, ETTm1, ETTm2, xxx]')
    parser.add_argument('--root_path', type=str, default='./dataset/', 
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.csv', 
                        help='data file')
    parser.add_argument('--features', type=str, default='M', 
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', 
                        help='target feature in S or MS task (OT is a column name); when features are set to M, this argument is ignored')
    parser.add_argument('--freq', type=str, default='h', 
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forcasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior of anomaly ratio')

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
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: {args.gpu}')
    else:
        args.device = torch.device('cpu')
        print('Using CPU')
    print('Arguments in experiment:')
    print_args(args)
    
    if args.task_name == 'anomaly_detection' or args.task_name == 'fault_detection':
        Exp = Exp_fault_detection
    elif args.task_name == 'fault_diagnosis':
        pass
    elif args.task_name == 'MLfault_detection':
        Exp = Exp_MLfault_detection
    elif args.task_name == 'KAEfault_detection':
        Exp = Exp_KAEfault_detection
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = f'{args.task_name}_{args.model}_{args.data}_lr{args.learning_rate}_te{args.train_epochs}_sl{args.seq_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_ff{args.d_ff}_exptime{ii}' \
                if args.task_name in ['fault_detection', 'anomaly_detection', 'KAEfault_detection'] else f'{args.task_name}_{args.model}_{args.data}_cr{args.cum_ratio}_cf{args.conf_level}'

            print(f'>>>>>>>>start training: {setting}<<<<<<<<<<<<<')
            exp.train(settings=setting)

            print(f'>>>>>>>>start testing: {setting}<<<<<<<<<<<<')
            exp.test(settings=setting)
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        ii = 0
        setting = f'{args.task_name}_{args.model}_{args.data}_lr{args.learning_rate}_te{args.train_epochs}_sl{args.seq_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_ff{args.d_ff}_exptime{ii}'
        print(f'>>>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<')
        exp.test(settings=setting, test=1)
        torch.cuda.empty_cache()


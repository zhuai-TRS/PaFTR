import argparse
import torch
import numpy as np
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from exp.exp_main import Exp_Main


def main():
    parser = argparse.ArgumentParser(description='Deep Learning Framework for Traffic Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model_id')
    parser.add_argument('--model', type=str, required=True, default='former', help='model_name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  # fixed
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model config
    parser.add_argument('--use_revin', type=int, default=0, help='1: use revin or 0: no revin')

    #FreCrossNet & TQNet
    parser.add_argument('--cycle_len', type=int, default=24, help='cycle length')
    # FreCrossNet
    parser.add_argument('--threshold', type=float, default=24.5, help='threshold for fre')

    # TimeEmb
    parser.add_argument('--use_hour_index', type=int, default=1, help='1: use hour_index or 0: no use')
    parser.add_argument('--use_day_index', type=int, default=0, help='1: use day_index or 0: no use')
    parser.add_argument('--hour_length', type=int, default=24, help='embedding length of hour index')
    parser.add_argument('--day_length', type=int, default=7, help='embedding length of day index')

    # TimeXer & PatchTST
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # TimesNet
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    # PatchTST
    parser.add_argument('--stride', type=int, default=8, help='stride')

    # DLinear
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

    # Formers
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # seed
    parser.add_argument('--random_seed', type=int, default=2026, help='seed')
    # Ablation experiment
    parser.add_argument('--use_seq_cycle_complex', type=str, default='complex',
                        help='[seq, cycle, complex]')
    parser.add_argument('--fusion_type', type=str, default='freq',
                        help='[freq, time_add, time_concat]')
    parser.add_argument('--qkv', type=str, default='cfs',
                        help='[cfs, csf, fcs, fsc, sfc, scf]')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    if args.is_training:
        for ii in range(args.itr):
            Exp = Exp_Main
            setting = '{}_{}_{}_sl{}_pl{}_cycle{}_seed{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.cycle_len,
                fix_seed
            )

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        Exp = Exp_Main
        setting = '{}_{}_{}_sl{}_pl{}_cycle{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.cycle_len,
            fix_seed
        )

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
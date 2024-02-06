import argparse
from os import getcwd, mkdir
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any,Optional, Union


def get_args_parser():
    parser = argparse.ArgumentParser('Set DistilledBert', add_help=False)
    
    # data
    parser.add_argument('--dir', default='debug', type=str, help='directory of checkpoints')
    parser.add_argument('--path_train', default='data/processed/train.csv', type=str,
                        help="data path of train set")
    parser.add_argument('--path_val', default='data/processed/val.csv', type=str,
                        help="data path of val set")
    
    # network
    parser.add_argument('--max_len', default=130, type=int, help='max length of the tokenizer')
    parser.add_argument('--qlora', action='store_true', help='use qlora')
    parser.add_argument('--qlora_last_layer', action='store_true', help='apply qlora only to the last layer instead of all')
    parser.add_argument('--qlora_rank', default=32, type=int, help='rank used in qlora')
    parser.add_argument('--qlora_alpha', default=32, type=int, help='gain used in qlora')
    parser.add_argument('--use_gru', action='store_true', help='use a GRU Decoder')
    parser.add_argument('--use_mlp', action='store_true', help='use MLP')
    parser.add_argument('--use_ltn', action='store_true', help='use constrainted loss during training')
    parser.add_argument('--nesy_constr', type=int, default=1, choices=[0,1,2], help='constraints version to be employed')
    parser.add_argument('--aggr_p', default=2, type=int, help='aggregator p-mean norm  value used during universal quantification')
    parser.add_argument('--step_p', default=-1, type=int, help='aggregator p-mean step increase time used during universal quantification')
    parser.add_argument('--freeze_emb', action='store_true', help='freeze embedding')
    parser.add_argument('--load_ckpt', action='store_true', help='load checkpoint from \
                        the directory previously created for the current configuration')

    # config
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--score_level', default=5, type=int, help='level of scores')
    parser.add_argument('--device', default='cuda', type=str, help='device used to train')
    parser.add_argument('--model_name', default='distilbert-base-uncased', type=str, help='name of the encoder model')

    # hyperparam
    parser.add_argument('--n_epoch', default=30, type=int, help='number of epochs')
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--no_scheduler', action='store_true', help='do not use the scheduler')
    parser.add_argument('--use_linear_scheduler', action='store_true', help='use a linear scheduler')
    parser.add_argument('--c_lr_min', default=2e-5, type=float)
    parser.add_argument('--c_lr_max', default=2e-3, type=float)
    parser.add_argument('--margin', default=0.15, type=float)
    parser.add_argument("--cls_loss", default='BCE',type=str, choices=['CE','BCE','L1', 'MSE'])
    parser.add_argument('--clip_norm', default=0.1, type=float, help="max possible norm before clipping procedure")
    parser.add_argument('--step_epoch', default=2*998, type=int, help='number of step per epochs')
    parser.add_argument('--emb_weight', default=1, type=float, help='embedding loss weight')
    parser.add_argument('--score_weight', default=10, type=float, help='score loss weight')
    parser.add_argument('--nesy_weight', default=1, type=float, help='score loss weight')
    parser.add_argument('--p_syn', default=0.1, type=float, help='probability of changing POS in a phrase')
    parser.add_argument('--use_sgd', action='store_true', help='use SDG Optmizer')
    parser.add_argument('--use_step_p', action='store_true', help='use step scheduler for p-mean of ForAll')
    parser.add_argument('--use_lamb', action='store_true', help='use LAMB Optimizer')

    # technicality 
    parser.add_argument('--seed', default=23, type=int, help='seed')
    parser.add_argument('--no_track', action='store_true', help='disable experiment tracking')

    # range test
    parser.add_argument('--lr_range_test', action='store_true', help='Perform LR Range Test')


    # optuna
    parser.add_argument('--use_optuna', action='store_true', help='use optuna to select hyperparameters')
    parser.add_argument('--use_ax', action='store_true', help='use optuna to select hyperparameters')
    parser.add_argument('--ax_name', type=str, help='name of the experiment')
    parser.add_argument('--n_trial', default=10, type=int, help='number of trial per study')
    parser.add_argument('--sw_low_bound',  type=int, help='low bound for the score weight loss')
    parser.add_argument('--sw_high_bound',  type=int, help='high bound for the score weight loss')
    parser.add_argument('--ew_low_bound',  type=int, help='low bound for the embedding weight loss')
    parser.add_argument('--ew_high_bound',  type=int, help='high bound for the embedding weight loss')
    parser.add_argument('--nw_low_bound',  type=float, help='low bound for the nesy weight loss')
    parser.add_argument('--nw_high_bound',  type=float, help='high bound for the nesy weight loss')
    parser.add_argument('--optuna_job',  type=int, help='number of optuna jobs')
    parser.add_argument("--optuna_sampler", default='bayesian',type=str, choices=['normal','bayesian'])

    return parser


def save_ckpt(
    net: torch.nn.Module,
    epoch:int ,
    loss: int ,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.CyclicLR] = None,
    dir: str='', 
    torch_state='',
    save_best:bool=False
    ) -> None:

    
    torch.save({ 'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'lr': scheduler.state_dict() if scheduler is not None else None,
                    'torch_state': torch_state
                    },
               get_ckpt_dir('best' if save_best else epoch, dir)
    )    


def load_ckpt(
    ckpt_dir:str,
    net: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.CyclicLR] = None
    ) -> int:

    ckpt = torch.load(ckpt_dir)

    net.load_state_dict(ckpt['model_state_dict'])

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(['lr'])

    torch.set_rng_state(ckpt['torch_state'])

    epoch = ckpt['epoch']


    return epoch








def get_ckpt_dir(epoch:int, dir:str='' ) -> str:
    if  osp.exists(dir)==False:
        mkdir(osp.join(getcwd(), dir) )
    
    return osp.join(getcwd(), f'{dir}/ckpt_{epoch}', )


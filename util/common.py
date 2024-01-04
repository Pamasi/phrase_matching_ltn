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
    
    parser.add_argument('--path_train', default='data/processed/train.csv', type=str,
                        help="data path of train set")
    parser.add_argument('--path_val', default='data/processed/val.csv', type=str,
                        help="data path of val set")
    parser.add_argument('--max_len', default=120, type=int, help='max length of the tokenizer')
    parser.add_argument('--qlora', action='store_true', help='use qlora')
    parser.add_argument('--qlora_rank', default=1, type=int, help='rank used in qlora')
    parser.add_argument('--qlora_alpha', default=1, type=int, help='gain used in qlora')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--score_level', default=5, type=int, help='level of scores')
    parser.add_argument('--device', default='cuda', type=str, help='device used to train')
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--c_lr_min', default=5e-6, type=float)
    parser.add_argument('--c_lr_max', default=5e-5, type=float)
    parser.add_argument('--margin', default=0.15, type=float)
    parser.add_argument('--emb_weight', default=1, type=float, help='embedding loss weight')
    parser.add_argument('--score_weight', default=10, type=float, help='score loss weight')
    parser.add_argument('--n_epoch', default=30, type=int, help='number of epochs')
    parser.add_argument('--seed', default=23, type=int, help='seed')
    parser.add_argument('--p_syn', default=0.5, type=float, help='probability of changing POS in a phrase')
    parser.add_argument('--no_track', action='store_true', help='disable experiment tracking')
    parser.add_argument('--optuna', action='store_true', help='use optuna to select hyperparameters')
    parser.add_argument('--freeze_emb', action='store_true', help='freeze embedding')
    parser.add_argument('--use_mlp', action='store_true', help='use MLP')
    parser.add_argument('--dir', default='debug', type=str, help='directory of checkpoints')
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
                    'lr': scheduler.state_dict(),
                    'torch_state': torch_state
                    },
               get_ckpt_dir('best' if save_best else epoch, dir)
    )    

def get_ckpt_dir(epoch:int, dir:str='' ) -> str:
    if  osp.exists(dir)==False:
        mkdir(osp.join(getcwd(), dir) )
    
    return osp.join(getcwd(), f'{dir}/ckpt_{epoch}', )


import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set DistilledBert', add_help=False)
    
    parser.add_argument('--path_train', default='data/processed/train.csv', type=str,
                        help="data path of train set")
    parser.add_argument('--path_val', default='data/processed/val.csv', type=str,
                        help="data path of val set")
    parser.add_argument('--max_len', default=120, type=int, help='max length of the tokenizer')
    parser.add_argument('--qlora', action='store_true', help='use qlora')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--score_level', default=5, type=int, help='level of scores')
    parser.add_argument('--device', default='cuda', type=str, help='device used to train')
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--c_lr_min', default=5e-6, type=float)
    parser.add_argument('--c_lr_max', default=5e-5, type=float)
    parser.add_argument('--margin', default=0.15, type=float)
    parser.add_argument('--emb_weight', default=0.1, type=float, help='embedding loss weight')
    parser.add_argument('--n_epoch', default=30, type=int, help='number of epochs')
    parser.add_argument('--seed', default=23, type=int, help='seed')
    parser.add_argument('--no_track', action='store_true', help='disable experiment tracking')
    parser.add_argument('--freeze_emb', action='store_true', help='freeze embedding')
    return parser
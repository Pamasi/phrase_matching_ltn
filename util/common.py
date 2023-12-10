import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set DistilledBert', add_help=False)
    
    parser.add_argument('--path_train', default='data/processed/train.csv', type=str,
                        help="data path of train set")
    parser.add_argument('--path_val', default='data/processed/val.csv', type=str,
                        help="data path of val set")
    parser.add_argument('--max_len', default=200, type=int, help='max length of the tokenizer')
    parser.add_argument('--batch', default=2, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--score_level', default=5, type=int, help='level of scores')
    parser.add_argument('--device', default='cuda', type=str, help='device used to train')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--c_lr_min', default=1e-6, type=float)
    parser.add_argument('--c_lr_max', default=1e-4, type=float)
    parser.add_argument('--n_epoch', default=30, type=int, help='number of epochs')
    parser.add_argument('--seed', default=23, type=int, help='seed')
    return parser
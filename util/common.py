import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set DistilledBert', add_help=False)
    
    parser.add_argument('--path_train', default='data/processed/train.csv', type=str,
                        help="data path of train set")
    parser.add_argument('--path_val', default='data/processed/val.csv', type=str,
                        help="data path of val set")
    parser.add_argument('--max_len', default=200, type=int, help='max length of the tokenizer')

    return parser
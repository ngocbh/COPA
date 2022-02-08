import os
import argparse
from epts.ept_config import EptConfig
from epts.ept_1 import run_ept_1
from epts.ept_2 import run_ept_2
from epts.ept_3 import run_ept_3
from epts.pretrain import pretrain_classifiers
from utils.helpers import make_logger
import logging
import matplotlib
import numpy as np
import torch
import random

logging.getLogger('matplotlib').setLevel(logging.ERROR)

all_datasets = ['synthetic_data', 'adult_data']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiments')
    parser.add_argument('--run-id', '-rid', default=0, type=int)
    parser.add_argument('--ept', '-e', dest='epts',
                        action='append', required=True)
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--datasets', dest='datasets', nargs='*')
    parser.add_argument('--methods', dest='methods', nargs='*')
    parser.add_argument('--classifiers', dest='classifiers', nargs='*')
    parser.add_argument('--num-proc', '-np', default=1, type=int)
    parser.add_argument('--update-config', '-uc', action='store_true')
    parser.add_argument('--plot-only', '-po', action='store_true')
    parser.add_argument('--seed', '-s', default=124, type=int)

    args = parser.parse_args()

    save_dir = f'results/run_{args.run_id}'
    config_path = os.path.join(save_dir, 'config.yml')

    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(args.seed - 1)
    torch.manual_seed(args.seed - 2)
    random.seed(args.seed - 3)
    np.set_printoptions(suppress=True)

    ec = EptConfig()
    # update config if needed
    if args.update_config or not os.path.isfile(config_path):
        ec.to_file(config_path, mode='merge_cls')
    else:
        ec.from_file(config_path)

    for ept in set(args.epts):
        ept_dir = os.path.join(save_dir, f'ept_{ept}')
        os.makedirs(ept_dir, exist_ok=True)
        logger = make_logger(f'ept_{ept}', ept_dir)

        if ept == "1":
            run_ept_1(ec, ept_dir, datasets=args.datasets,
                      classifiers=args.classifiers, num_proc=args.num_proc,
                      plot_only=args.plot_only, seed=args.seed, logger=logger)
        elif ept == "2":
            run_ept_2(ec, ept_dir, datasets=args.datasets, methods=args.methods,
                      classifiers=args.classifiers, num_proc=args.num_proc,
                      plot_only=args.plot_only, seed=args.seed, logger=logger)
        elif ept == "3":
            run_ept_3(ec, ept_dir, datasets=args.datasets, methods=args.methods,
                      classifiers=args.classifiers, num_proc=args.num_proc,
                      plot_only=args.plot_only, seed=args.seed, logger=logger)
        elif ept == "pretrain":
            pretrain_classifiers(ec, ept_dir, datasets=args.datasets,
                                 classifiers=args.classifiers,
                                 num_proc=args.num_proc, logger=logger)

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from src.core import YAMLConfig
from src.solver import DetSolver
import warnings
import src.misc.dist as dist

warnings.filterwarnings('ignore')

def main(args, ) -> None:
    dist.init_distributed_mode(args)
    cfg = YAMLConfig(args)
    print(cfg.config.get('model'))
    solver = DetSolver(cfg)
    if args.mode == 'train':
        solver.fit()
    else:
        solver.val()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default='train', choices=['train', 'val'], type=str)
    parser.add_argument('--yaml_path', '-c', default='cfg/models/X3D18_4816.yaml', type=str)
    parser.add_argument('--use_amp', default=True, type=bool)
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )

    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--rank', default=0, type=int)
    args = parser.parse_args()
    main(args)

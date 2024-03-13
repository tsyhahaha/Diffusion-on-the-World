import random
import numpy as np
import argparse
from math import sqrt
from common import sample_rot_2d, sample_tr_2d, visual_2d

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=100)
    parser.add_argument('--nodes', type=int, default=3)
    parser.add_argument('--output', type=str, default='./output.npy')
    return parser.parse_args()

def main(args):
    base_tri = np.array([(1, 0), (-1, 0), (0, sqrt(3))])
    base_tri = base_tri - np.mean(base_tri, axis=0)
    rots = sample_rot_2d(args.number)
    trs = sample_tr_2d(args.number)

    base_tri = base_tri[:, :, None]
    rots = np.array(rots)[:, None, :, :]
    trs = np.array(trs)[:, None, :]
    transformed_tri = np.matmul(rots, base_tri).squeeze()+ trs
    # visual_2d(transformed_tri)
    np.save(args.output, transformed_tri)


if __name__=='__main__':
    args = parse()
    main(args)
    

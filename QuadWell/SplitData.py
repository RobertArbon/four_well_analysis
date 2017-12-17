import numpy as np
from os.path import join
import sys


def split_and_save(tau, in_dir, out_dir, fname, offset):
    x = np.load(join(in_dir, fname))
    for i in range(x.shape[0]-tau):
        out = np.zeros((2, 1))
        out[[0, 1], :] = x[[i,i+tau],:]
        np.save(join(out_dir, 'traj-{:06d}.npy'.format(offset*(x.shape[0]-tau)+i)), out)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    files = ['quad_well_{:02d}.npy'.format(x) for x in range(100)]

    tau = 25
    print('Getting trajectories from {}'.format(in_dir))
    print('Saving split trajectories to {0}, split with tau = {1}\n\n'.format(out_dir, tau))
    for j, fname in enumerate(files):
        print(fname, end=', ')
        split_and_save(tau, in_dir, out_dir, fname, j)
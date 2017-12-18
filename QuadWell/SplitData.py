import numpy as np
from os.path import join
import sys
from glob import glob

# def split_and_save(tau, in_dir, out_dir, fname, offset):
#     x = np.load(join(in_dir, fname))
#     for i in range(x.shape[0]-tau):
#         out = np.zeros((2, 1))
#         out[[0, 1], :] = x[[i,i+tau],:]
#         np.save(join(out_dir, 'traj-{:06d}.npy'.format(offset*(x.shape[0]-tau)+i)), out)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    files = glob(join(in_dir, '*.npy'))

    tau = 25
    print('Getting trajectories from {}'.format(in_dir))
    print('Saving single split trajectory to {0}, split with tau = {1}\n'.format(out_dir, tau))

    all_X = [np.load(x) for x in files]

    # There's a lot better way of doing this.
    X_split = []
    for X in all_X:
        for i in range(X.shape[0] - tau):
            X_split.append(list(*X[[i, i + tau], :].T))

    X_obs = np.array([x.shape[0] for x in all_X])
    X_pairs = np.array([x-tau for x in X_obs])

    X_split = np.array(X_split)
    print('Total number of trajectories: {}'.format(len(all_X)))
    print('Total number of observations: {}'.format(np.sum(X_obs)))
    print('Total number of pairs       : {}'.format(np.sum(X_pairs)))
    print('X_split shape               : {}'.format(X_split.shape))
    np.save(join(out_dir, 'X-{}.npy'.format(tau)), X_split)

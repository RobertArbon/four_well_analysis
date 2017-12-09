import numpy as np
from os.path import exists, join, abspath
from os import makedirs
from glob import glob
"""
filters data

"""

data_dir = abspath(join('.', 'data', '100.0pc'))
fnames = ['quad_well_{:02}.npy'.format(i) for i in range(100)]
X = {}
for fname in fnames:
    X[fname] = np.load(join(data_dir, fname))

fractions = np.logspace(np.log10(0.005),np.log10(1),5)

for fraction in fractions[:-1]:
    directory = abspath(join('.', 'data','{:05.1f}pc'.format(fraction*100)))
    print(directory)
    if not exists(directory):
        makedirs(directory)
    for k, v in X.items():
        idx = int(v.shape[0]*fraction)
        np.save(join(directory, k), v[:idx, :])

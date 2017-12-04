from sklearn.base import BaseEstimator
from pyemma.msm import MaximumLikelihoodMSM, BayesianMSM
from msmbuilder.cluster import NDGrid
from sklearn.pipeline import Pipeline
from glob import glob
import numpy as np


class DataFilter(BaseEstimator):
    """
    Filters the amount of data entering the next stage of pipeline.
    Assumes you filter over the first axis
    """

    def __init__(self, mode='initial', fraction=1.0):
        if mode not in ['initial']:
            raise NotImplementedError("Only 'initial' filter implemented")
        else:
            self.mode = mode
        if fraction > 1.0 or fraction <= 0.0:
            raise ValueError("'fraction' must be in range (0,1]")
        else:
            self.fraction = fraction
        self.lengths = []

    def fit(self, X, y=None):
        if not isinstance(X, list):
            raise TypeError('X must be of list type')
        self.lengths = [int(x.shape[0]*self.fraction) for x in X]

    def predict(self, X):
        if self.mode == 'initial':
            return [x[:i] for x, i in zip(*[X, self.lengths])]


traj_paths = glob('data/*.npy')
X = [np.load(traj_path) for traj_path in traj_paths]

xmin, xmax = -1.2, 1.2
tau = 25

model = Pipeline([('filter', DataFilter(fraction=0.05)),
                  ('cluster',NDGrid(min=xmin, max=xmax, n_bins_per_feature=200)),
                  ('msm', MaximumLikelihoodMSM(lag=tau))])
model.fit(X)



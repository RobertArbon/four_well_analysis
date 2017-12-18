from pyemma.msm import MaximumLikelihoodMSM
from pyemma.coordinates.util import DtrajReshape
from msmbuilder.cluster import NDGrid
from sklearn.pipeline import Pipeline
import pickle
from glob import glob
import numpy as np

# traj_paths = glob('data/000.5pc/*.npy')
# X = [np.load(traj_path) for traj_path in traj_paths]

xmin, xmax = -1.2, 1.2
tau = 25

model = Pipeline([('reshape',  DtrajReshape()),
                  ('cluster',NDGrid(min=xmin, max=xmax, n_bins_per_feature=200)),
                  ('msm', MaximumLikelihoodMSM(lag=1, score_method='vamp1'))])


pickle.dump(model, open('model_lag1.pickl', 'wb'))




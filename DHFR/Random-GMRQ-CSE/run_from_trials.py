"""
This takes and old database of trials and re-runs the various models using a different scoring method. 
The folds won't be the same but as we're only interested in average quantities that shouldn't be a problem. 
"""
from osprey.config import Config
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from msmbuilder.feature_selection import VarianceThreshold
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from os.path import join
from glob import glob
import numpy as np
from multiprocessing import Pool
import pandas as pd

# Globals
config_path = '../../Trial Data/DHFR/Random-GMRQ-2/alpha_angle.yaml'
db_path = '../../Trial Data/DHFR/Random-GMRQ-2/osprey-trials.db'
traj_dir = '/home/robert/Datasets/DHFR/train'


def get_pipeline(parameters):
    """
    Wrapper so that new instance of a pipeline can be instantiated for every fold. 
    :return: sklean.pipeline.Pipeline object
    """
    pipe = Pipeline([('variance_cut', VarianceThreshold()),
                     ('tica', tICA(kinetic_mapping=True)),
                     ('cluster', MiniBatchKMeans()),
                     ('msm', MarkovStateModel(use_gap='timescales', lag_time=50, verbose=True))])
    pipe.set_params(**parameters)

    return pipe


# cross validation iterator
# TODO get this from the config file
# cv:
#     name: shufflesplit
#     params:
#       n_splits: 5
#       test_size: 0.5
cv = ShuffleSplit(n_splits=2, test_size=0.5)


def get_trajectories(feat):
    """
    Gets the trajctories associated with a feature
    :param feat: 
    :return: 
    """
    traj_paths = glob(join(traj_dir, feat, '*'))
    trajs = [np.load(traj_path) for traj_path in traj_paths]
    return trajs


def get_parameters(irow):
    """
    Gets the parameters for running a new model. 
    :return: dictionary of parameters
    """
    i, row = irow
    params_dict = row['parameters']

    params = {}
    trial_config = {}

    params['tica__lag_time'] = params_dict['tica__lag_time']
    params['tica__n_components'] = params_dict['tica__n_components']
    params['cluster__n_clusters'] = params_dict['cluster__n_clusters']

    trial_config['params'] = params
    trial_config['feature'] = row['project_name']
    trial_config['id'] = row['id']

    return trial_config


def run_trial(trial_config):

    X = get_trajectories(trial_config['feature'])
    id = trial_config['id']

    train_scores = []
    train_gaps = []
    train_n_timescales = []
    test_scores = []

    for train_idx, test_idx in cv.split(X):
        pipe = get_pipeline(trial_config['params'])

        train = [X[idx] for idx in train_idx]
        pipe.fit(train)

        train_n_timescales.append(pipe.named_steps['msm'].n_timescales)
        train_gaps.append(pipe.named_steps['msm'].gap_)
        train_scores.append(pipe.score(train))

        test = [X[idx] for idx in test_idx]
        try:
            score = pipe.score(test)
        except:
            score = None
        test_scores.append(score)

    results = {'id': id, 'cse_train_scores': train_scores, 'cse_train_gaps': train_gaps,
               'cse_train_n_timescales': train_n_timescales, 'cse_test_scores': test_scores }
    return results


if __name__ == "__main__":
    np.random.seed(42)
    config = Config(config_path)
    trials = config.trial_results()
    trials = trials.iloc[:2, :]
    trial_configs = [get_parameters(irow) for irow in trials.iterrows()]

    with Pool(4) as pool:
        results = list(pool.imap_unordered(run_trial, trial_configs))

    data = {'id': [x['id'] for x in results],
            'cse_train_scores': [x['cse_train_scores'] for x in results],
            'cse_train_gaps': [x['cse_train_gaps'] for x in results],
            'cse_train_n_timescales': [x['cse_train_n_timescales'] for x in results],
            'cse_test_scores': [x['cse_test_scores'] for x in results]}
    df2 = pd.DataFrame(data=data)
    all_trials = trials.merge(right=df2, how='outer', on='id')

    pd.to_pickle(all_trials, 'cse_trials.pickl')






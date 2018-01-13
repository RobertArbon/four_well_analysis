from pyemma.msm import MaximumLikelihoodMSM
from msmbuilder.cluster import NDGrid
from sklearn.pipeline import Pipeline
import pickle
import sys

if __name__ == '__main__':

    xmin, xmax = -1.2, 1.2
    tau = 25

    k = int(sys.argv[1])

    model = Pipeline([('cluster',NDGrid(min=xmin, max=xmax, n_bins_per_feature=200)),
                      ('msm', MaximumLikelihoodMSM(lag=tau, score_method='vampe', score_k=k))])


    pickle.dump(model, open('msm-k_{}.pickl'.format(k),'wb'))

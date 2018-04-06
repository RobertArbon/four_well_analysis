print('start')
from pyemma.msm import MaximumLikelihoodMSM, MaximumLikelihoodHMSM, estimate_hidden_markov_model
from msmbuilder.cluster import NDGrid
import pickle
from glob import glob
import numpy as np
from scipy.stats import entropy
import pandas as pd
import bhmm

def bic(hmm):
    p = dof(hmm)
    ndata = n_obs(hmm)
    loglike = hmm.likelihood
    return np.log(ndata) * p - 2 * loglike


def icl(hmm):
    return 2 * class_entropy(hmm) + bic(hmm)


def class_entropy(hmm):
    h_probs = hmm.hidden_state_probabilities
    ent = np.sum([np.sum(entropy(x.T)) for x in h_probs])
    return ent


def aic(hmm):
    p = dof(hmm)
    loglike = hmm.likelihood  # log likelihood
    return 2 * p - 2 * loglike


def n_obs(hmm):
    nobs = np.sum([x.shape[0] for x in hmm.dtrajs_obs])
    return nobs


def dof(hmm):
    if (hmm.separate is not None) or (hmm.connectivity is None) or \
                    hmm.connectivity == 'all':
        print("BIC/AIC not available for these constraints")
        print('Is separate: ', hmm.separate)
        print('Connectivity: ', hmm.connectivity)
        return None
    else:
        N = hmm.metastable_distributions.shape[0]  # Num hidden states
        M = hmm.metastable_distributions.shape[1]  # Num observed states

        dof = N * (M - 1)  # Emission probabilities add to one.
        if hmm.reversible:
            dof += (1 / 2) * N * (N - 1) + N - 1
        else:
            dof += N * (N - 1)
        dof = int(dof)

        return dof


def get_taus(ts):
    grouped_ts = []
    i = 0
    while i < len(ts) - 1:
        if np.abs(ts[i + 1] - ts[i]) < 2:
            grouped_ts.append(int(np.mean(ts[i:i + 2])))
            i += 2
        else:
            grouped_ts.append(int(ts[i]))
            i += 1
    taus = [int((grouped_ts[i]+grouped_ts[i+1])/2) for i in range(len(grouped_ts)-1)]
    return taus


def get_ergodic_set(dtrajs, tau):

    m = MaximumLikelihoodMSM(lag=tau, connectivity='largest', reversible=True)
    m.fit(dtrajs)

    erg_dtrajs = [x for x in m.dtrajs_active if not -1 in x]

    m = MaximumLikelihoodMSM(lag=tau, connectivity='largest', reversible=True)
    m.fit(erg_dtrajs)

    assert m.active_count_fraction == 1.0, 'Active count fraction not 1'

    return erg_dtrajs


def get_dtrajs(X, xmin, xmax, m):
    cluster = NDGrid(min=xmin, max=xmax, n_bins_per_feature=m)
    dtrajs = cluster.fit_transform(X)
    return dtrajs


if __name__ == '__main__':
    print('in main')
    data_name = 'four-well-long'
    X = [np.load(x) for x in glob('Data/' + data_name + '/*npy')]
    xmin = np.min(np.concatenate(X))
    xmax = np.max(np.concatenate(X))

    ts = np.load('timescales.npy')
    taus = get_taus(ts)

    dtrajs = get_dtrajs(X, xmin=xmin, xmax=xmax, m=200)
    
    erg_dtrajs = get_ergodic_set(dtrajs=dtrajs, tau=taus[0])
    
    ks = np.arange(2,10)
    results = {'tau': [], 'k': [], 'bic': [], 'aic': [], 'icl': [], 'entropy': [], 'n_obs': [], 'dofs': []}
    for tau in taus:

        print('tau = ', tau)

        for k in ks:
            print('\tk = ', k)

            m = MaximumLikelihoodMSM(lag=tau, connectivity='largest', reversible=True)
            m.fit(erg_dtrajs)

            assert m.active_count_fraction == 1.0, 'Active count fraction not 1.0'

            print('\tFitting HMM')

            hmm = estimate_hidden_markov_model(dtrajs=erg_dtrajs[:2], nstates=int(k), lag=tau, stationary=False, reversible=True, connectivity='largest')
            #hmm = estimate_hidden_markov_model(dtrajs=erg_dtrajs[:2], nstates=2, lag=2) #, stationary=False, reversible=True, connectivity='largest')
            #hmm = MaximumLikelihoodHMSM(nstates=int(k), lag=tau, stationary=False, reversible=True, connectivity='largest', msm_init=m)
            #hmm.fit(erg_dtrajs[:2])

            results['k'].append(k)
            results['tau'].append(tau)
            results['bic'].append(bic(hmm))
            results['aic'].append(aic(hmm))
            results['icl'].append(icl(hmm))
            results['entropy'].append(class_entropy(hmm))
            results['dofs'].append(dof(hmm))
            results['n_obs'].append(n_obs(hmm))

    pd.DataFrame(results).to_pickle('h_state_selection.p')




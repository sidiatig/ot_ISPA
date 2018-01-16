import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as fucs

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import numpy as np
import time
import ot

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


def gromov_barycenters(N, Xs, ps, p, lambdas, loss_fun, epsilon,
                       max_iter=1000, tol=1e-9):

    S = len(Xs)
    Xs = [np.asarray(Xs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    xc = 1 / S * sum(Xs[s] for s in range(S))

    cpt = 0
    err = 1

    while(err > tol and cpt < max_iter):

        Cprev = C
        M = [ot.utils.dist(Xs[s],xc) for s in range(S)]
        Ts = [ot.da.sinkhorn_l1l2_gl(Xs[s], ys[s], xc, M[s], reg, eta) for s in range(S)]
        if loss_fun == 'square_loss':
            xc = ot.gromov.update_square_loss(p, lambdas, Ts, Xs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1
    print('number of cpt in gromov', cpt)
    if log:
        return C, log
    else:
        return C
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
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy as sp
import numpy as np
import time

import ot

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'
    experiment = 'fmril'
    nor_method = 'no'
    clf_method = 'svm'
    source_id = np.arange(5)
    target_id = 6
    op_function = 'l1l2' # 'lpl1' 'l1l2'

    if experiment == 'fmril':
        source = fmril
    elif experiment == 'fmrir':
        source = fmrir
    else:
        source = meg

    result_dir = result_dir + '/{}'.format(experiment)
    y_target = source.y_target
    subjects = np.array(source.subjects)
    x_data = source.x_data_pow if experiment == 'meg' else source.x_data
    x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
    x = x_indi if nor_method == 'indi' else x_data

    indices_sub = fucs.split_subjects(subjects)

    S = len(source_id)
    xs = [x[indices_sub[i]] for i in source_id]
    ys = [y_target[indices_sub[i]] for i in source_id]
    xt = x[indices_sub[target_id]]
    yt = y_target[indices_sub[target_id]]
    Cs = [sp.spatial.distance.cdist(xs[s], xs[s]) for s in range(S)]
    Cs = [cs / cs.max() for cs in Cs]
    ns = [len(xs[s]) for s in range(S)]
    ps = [ot.unif(ns[s]) for s in range(S)]
    p = ot.unif(len(xt))
    lambdas = ot.unif(S)
    C, T = ot.gromov.gromov_barycenters(len(xt), Cs, ps, p, lambdas,
                                           'square_loss', 5e-4,
                                           max_iter=100, tol=1e-5)
    ot_source = []

    note = 'barycenter_{}s_{}t_{}_{}_{}_{}'.format(source_id, target_id, experiment,
                                                 nor_method, clf_method, op_function)
    print(note)




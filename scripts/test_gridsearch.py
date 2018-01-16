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
import scipy.stats as stats
import numpy as np
import time

import ot


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

def get_source(n,m,nz=0.5):
    y = np.floor((np.arange(n) * 1.0 / n * 2))+1
    x = np.zeros((n, m))
    # class 1
    x[y == 1] = np.ones(m)
    x[y == 2] = np.concatenate((np.zeros(int(m/2)), np.ones(int(m/2))))
    print(x[0])
    x[y != 2, :] += 1.5 * nz * np.random.randn(sum(y != 2), m)
    x[y == 2, :] += 2 * nz * np.random.randn(sum(y == 2), m)
    print(x[0])
    return x,y.astype(int)
def get_target(n, m, nz=0.5):
    y = np.floor((np.arange(n) * 1.0 / n * 2))+1
    x = np.zeros((n, m))
    x[y == 1] = np.ones(m) * -2
    x[y == 2] = np.concatenate((np.zeros(int(m / 2)), np.ones(int(m / 2))*2))
    print(x[0])
    x[y != 2, :] += nz * np.random.randn(sum(y != 2), m)
    x[y == 2, :] += 2 * nz * np.random.randn(sum(y == 2), m)
    print(x[0])
    return x, y.astype(int)

def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'

    # args = sys.argv[1:]
    # experiment = args[0]
    # nor_method = args[1]
    # clf_method = args[2]
    # source_id = int(args[3])
    # target_id = int(args[4])
    # op_function = args[5] # 'lpl1' 'l1l2'

    metric_xst = 'h'

    experiment = 'fmrir'
    nor_method = 'no'
    clf_method = 'logis'
    source_id = 1
    target_id = 2
    op_function = 'l1l2'  # 'lpl1' 'l1l2'

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
    nb_subs = len(indices_sub)

    i = source_id
    j = target_id
    xs = x[indices_sub[i]]
    ys = y_target[indices_sub[i]]
    xt = x[indices_sub[j]]
    yt = y_target[indices_sub[j]]
    note = '{}s_{}t_{}_{}_{}_sinkhorn_{}_{}'.format(i, j, experiment,
                                                    nor_method, clf_method, op_function, metric_xst)
    accs = fucs.get_accuracy(xs,ys,xt,yt,clf_method='logis')
    print(accs)
if __name__ == '__main__':
    main()


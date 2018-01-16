import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as fucs

import numpy as np
import time
import ot


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'

    experiment = 'fmril'
    nor_method = 'indi'
    clf_method = 'svm'

    op_function = 'maplin' # 'lpl1' 'l1l2'

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


    sub_ids = np.arange(5)
    xs1 = [x_data[indices_sub[i]][:20] for i in sub_ids]
    xs2 = [x_data[indices_sub[i]][20:] for i in sub_ids]
    xs_indi1 = [x_indi[indices_sub[i]][:20] for i in sub_ids]
    xs_indi2 = [x_indi[indices_sub[i]][20:] for i in sub_ids]

    def barycenter(Xs, loss_fun, reg, max_iter, tol):
        S = len(Xs)
        Xs = [np.asarray(Xs[s], dtype=np.float64) for s in range(S)]
        ps = [ot.utils.unif(Xs[s].shape[0]) for s in range(S)]
        p = ot.utils.unif(Xs[0].shape[0])
        Cs = [ot.utils.dist(Xs[s], Xs[s], metric='euclidean') for s in range(S)]
        Cs = [cs / np.max(cs) for cs in Cs]
        weights = ot.utils.unif(S)
        C = ot.gromov.gromov_barycenters(Xs[0].shape[0], Cs,
                                             ps, p, weights, loss_fun, reg,
                                             max_iter=max_iter, tol=tol, log=False)
        return C
    reg = 0.1
    C1 = barycenter(xs_indi1, 'square_loss', reg, 100, 1e-5)
    C2 = barycenter(xs_indi2, 'square_loss', reg, 100, 1e-5)

    xt_indi = x_indi[indices_sub[8]]
    p1 = ot.utils.unif(xs_indi1[0].shape[0])
    p_t = ot.utils.unif(xs_indi1[0].shape[0])

    data1 = xt_indi[:20]
    data2 = xt_indi[20:]
    c_t1 = ot.utils.dist(data1, data1, metric='euclidean')
    eta = 1
    gw_distance1 = ot.gromov.gromov_wasserstein2(c_t1, C1, p_t, p1, 'square_loss', eta, 100, 1e-5)
    gw_distance2 = ot.gromov.gromov_wasserstein2(c_t1, C2, p_t, p1, 'square_loss', eta, 100, 1e-5)
    print(gw_distance1, gw_distance2)
    c_t2 = ot.utils.dist(data2, data2, metric='euclidean')
    gw_distance1 = ot.gromov.gromov_wasserstein2(c_t2, C1, p_t, p1, 'square_loss', eta, 100, 1e-5)
    gw_distance2 = ot.gromov.gromov_wasserstein2(c_t2, C2, p_t, p1, 'square_loss', eta, 100, 1e-5)
    print(gw_distance1, gw_distance2)



if __name__ == '__main__':
    main()
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
import numpy as np
import time

import ot

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'
    op_function='l1l2'
    clf_method = 'logis'
    n_samples_source = 150
    n_samples_target = 150

    xs, ys = ot.datasets.get_data_classif('3gauss', n_samples_source)
    xt, yt = ot.datasets.get_data_classif('3gauss2', n_samples_target)

    # Cost matrix
    M = ot.dist(xs, xt, metric='sqeuclidean')

    scores_params = {'source': [], 'target': [], 'ori_score': [], 'ot_score': [],
                         'params': []}
    unsec = 0
    # target_ids = [j for j in range(nb_subs) if j != i]

    # cityblock sqeuclidian, minkowski

    best_xst, ot_params = fucs.pairsubs_sinkhorn_lables(xs,ys,xt,
                                                        ot_method=op_function,
                                                        metric='h')

    # train on xst, test on xt
    ot_score = fucs.get_accuracy(best_xst,ys,xt,yt,
                                clf_method=clf_method)
    print('ot source data predicts on original target, accuracy', ot_score)
    # print('ot source data best params', ot_best)
    # train on xs, test on xt
    ori_score = fucs.get_accuracy(xs, ys, xt, yt, clf_method=clf_method)
    print('original source data predicts on original target, accuracy', ori_score)
    # print('original source data best params', ori_best)


    pair_params = {}
    pair_params['source'] = 1
    pair_params['target'] = 2
    pair_params['ori_score'] = round(ori_score, 4)
    pair_params['ot_score'] = round(ot_score, 4)
    pair_params['params'] = ot_params

    if ori_score >= ot_score:
        unsec += 1
        # joblib.dump(pair_params, result_dir+'/gridtables/{}_unsuccessful_ot.pkl'.format(note))
    print('-----------------------------------------------------------------------------------------')
    print(pair_params)
    # # note = '{}s_{}ts_{}_{}_{}_sinkhorn_{}_{}'.format(i, len(target_ids), experiment,
    # #                                                 nor_method, clf_method, op_function, metric_xst)
    # # joblib.dump(scores_params, result_dir + '/{}_score_params.pkl'.format(note))
    # print('unsuccessful transport', unsec)
    # print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

if __name__ == '__main__':
    main()






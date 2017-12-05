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


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'

    args = sys.argv[1:]
    experiment = args[0]
    nor_method = args[1]
    clf_method = args[2]
    source_id = int(args[3])
    target_id = int(args[4])
    op_function = args[5] # 'lpl1' 'l1l2'

    metric_xst = 'h'

    # experiment = 'fmrir'
    # nor_method = 'indi'
    # clf_method = 'logis'
    # source_id = 1
    # op_function = 'l1l2' # 'lpl1' 'l1l2'


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
    xs = x[indices_sub[i]]
    ys = y_target[indices_sub[i]]


    # scores_params = {'source': [], 'target': [], 'ori_score': [], 'ot_score': [],
    #                      'params': []}
    # unsec = 0
    # target_ids = [j for j in range(nb_subs) if j != i]
    # target_ids = [2]
    if True:
        j = target_id
        note = '{}s_{}t_{}_{}_{}_sinkhorn_{}_{}'.format(i, j, experiment,
                                                     nor_method, clf_method, op_function,metric_xst)
        print(note)
        xt = x[indices_sub[j]]
        yt = y_target[indices_sub[j]]
        # cityblock sqeuclidian, minkowski

        best_xst, ot_params = fucs.pairsubs_sinkhorn_lables(xs,ys,xt,
                                                            ot_method=op_function,
                                                            metric=metric_xst)

        # train on xst, test on xt
        ot_score = fucs.get_accuracy(best_xst,ys,xt,yt,
                                    clf_method=clf_method)
        print('ot source data predicts on original target, accuracy', ot_score)
        # print('ot source data best params', ot_best)
        # train on xs, test on xt
        ori_score = fucs.get_accuracy(xs, ys, xt, yt, clf_method=clf_method)
        print('original source data predicts on original target, accuracy', ori_score)
        # print('original source data best params', ori_best)
        # scores_params['source'].append(i)
        # scores_params['target'].append(j)
        # scores_params['ori_score'].append(round(ori_score,4))
        # scores_params['ot_score'].append(round(ot_score,4))
        # scores_params['params'].append(ot_params)

        pair_params = {}
        pair_params['source'] = i
        pair_params['target'] = j
        pair_params['ori_score'] = round(ori_score, 4)
        pair_params['ot_score'] = round(ot_score, 4)
        pair_params['params'] = ot_params
        print('pair_params',pair_params)
        if ori_score >= ot_score:
            # unsec += 1
            joblib.dump(pair_params, result_dir+'/gridtables/{}_unsuccessful_ot.pkl'.format(note))
        joblib.dump(pair_params, result_dir + '/{}_score_params.pkl'.format(note))
        print('-----------------------------------------------------------------------------------------')
    # print(scores_params)
    # note = '{}s_{}ts_{}_{}_{}_sinkhorn_{}_{}'.format(i, len(target_ids), experiment,
    #                                                 nor_method, clf_method, op_function, metric_xst)
    # joblib.dump(scores_params, result_dir + '/{}_score_params.pkl'.format(note))
    # print('unsuccessful transport', unsec)
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

if __name__ == '__main__':
    main()






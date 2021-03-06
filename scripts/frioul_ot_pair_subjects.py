import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from data import fmri_localizer as newfmri
from model import procedure_function as fucs
import matplotlib
matplotlib.use('Agg')
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import time



print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

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

    metric_xst = "h"

    experiment = 'newfmri'
    nor_method = 'no'
    clf_method = 'svm'
    source_id = 1
    target_id = 2
    op_function = 'l1l2' # 'lpl1' 'l1l2'

    if experiment == 'fmril':
        source = fmril
    elif experiment == 'fmrir':
        source = fmrir
    elif experiment == 'meg':
        source = meg
    else:
        source = newfmri

    result_dir = result_dir + '/{}'.format(experiment)
    y_target = source.y_target
    subjects = np.array(source.subjects)
    x_data = source.x_data_pow if experiment == 'meg' else source.x_data
    x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
    x = x_indi if nor_method == 'indi' else x_data

    indices_sub = fucs.split_subjects(subjects)
    nb_subs = len(indices_sub)
    pca = PCA(n_components=144)
    i = source_id
    j = target_id
    xs = x[indices_sub[i]]
    xs_pca = pca.fit_transform(xs)
    ys = y_target[indices_sub[i]]
    xt = x[indices_sub[j]]
    xt_pca = pca.fit_transform(xt)
    yt = y_target[indices_sub[j]]
    note = 'new_pca_h_{}s_{}t_{}_{}_{}_{}'.format(i, j, experiment,
                                                 nor_method, clf_method, op_function)
    print(note)

    # cityblock sqeuclidian, minkowski
    best_xst, ot_params, params_acc = fucs.pairsubs_sinkhorn_lables_h(xs, ys, xt,
                                                                      ot_method=op_function, clf_method=clf_method)

    # train on xst, test on xt
    ot_accs = fucs.get_accuracy(best_xst, ys, xt, yt, clf_method=clf_method)
    ot_score = np.mean(ot_accs)
    print('ot source data predicts on original target, accuracy', ot_score, ot_accs)

    # train on xs, test on xt
    ori_accs = fucs.get_accuracy(xs, ys, xt, yt, clf_method=clf_method)
    ori_score = np.mean(ori_accs)
    print('original source data predicts on original target, accuracy', ori_score, ori_accs)
    # train on xs indi-nor, test on xt indi-nor
    x_base = x_indi.copy()
    xs_base = x_base[indices_sub[i]]
    ys_base = y_target[indices_sub[i]]
    xt_base = x_base[indices_sub[j]]
    yt_base = y_target[indices_sub[j]]
    base_accs = fucs.get_accuracy(xs_base, ys_base, xt_base, yt_base, clf_method=clf_method)
    base_score = np.mean(base_accs)
    print('base: indi-normalized source predicts on indi-normalized target, accuracy', base_score, base_accs)

    pair_params = {}
    pair_params['source'] = i
    pair_params['target'] = j
    pair_params['ori_score'] = round(ori_score, 4)
    pair_params['ori_accs'] = ori_accs
    pair_params['ot_accs'] = ot_accs
    pair_params['ot_score'] = round(ot_score, 4)
    pair_params['base_accs'] = base_accs
    pair_params['base_score'] = round(base_score, 4)
    pair_params['params'] = ot_params
    pair_params['params_acc'] = params_acc
    ttest_base_ot = stats.ttest_rel(base_accs,ot_accs)
    pair_params['ttest'] = (ttest_base_ot.pvalue, ttest_base_ot.statistic)
    print('pair_params', pair_params)
    ttest_ori_ot = stats.ttest_rel(ori_accs, ot_accs)
    pair_params['ttest_ori_ot'] = (ttest_ori_ot.pvalue, ttest_ori_ot.statistic)
    if ttest_base_ot.statistic > 0:
        joblib.dump(pair_params, result_dir + '/big_base/{}_pair_params.pkl'.format(note))
    else:
        joblib.dump(pair_params, result_dir + '/small_base/{}_pair_params.pkl'.format(note))
    if ttest_ori_ot.statistic > 0:
        joblib.dump(pair_params, result_dir + '/big_ori/{}_pair_params.pkl'.format(note))
    else:
        joblib.dump(pair_params, result_dir + '/small_ori/{}_pair_params.pkl'.format(note))
    reg_coor = np.log10(params_acc['params'])
    zero_coor = reg_coor[params_acc['acc'] == 0]
    other_coor = reg_coor[params_acc['acc'] != 0]
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot(zero_coor[:, 0], zero_coor[:, 1], '+r', label='acc is 0.5')
    plt.plot(other_coor[:, 0], other_coor[:, 1], 'ob', label='acc not 0.5')
    plt.legend(loc=0)
    plt.title(note)

    plt.subplot(1, 2, 2)
    accs = params_acc['acc'].reshape((6, 6))
    accs2 = np.empty((accs.T.shape))
    for k in range(accs.shape[0]):
        accs2[:, k] = accs[k][::-1]

    plt.imshow(accs2)
    plt.colorbar()
    plt.title('imshow')
    plt.savefig(main_dir + '/figs/acc_imshow_{}.png'.format(note))
    print('-----------------------------------------------------------------------------------------')
    # print(scores_params)
    # note = '{}s_{}ts_{}_{}_{}_sinkhorn_{}_{}'.format(i, len(target_ids), experiment,
    #                                                 nor_method, clf_method, op_function, metric_xst)
    # joblib.dump(scores_params, result_dir + '/{}_score_params.pkl'.format(note))
    # print('unsuccessful transport', unsec)
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

if __name__ == '__main__':
    main()






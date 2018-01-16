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

def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'

    # args = sys.argv[1:]
    # experiment = args[0]
    # nor_method = args[1]
    # clf_method = args[2]
    # source_id = int(args[3])
    # target_id = int(args[4])
    # op_function = args[5] # 'lpl1' 'l1l2' 'maplin'



    experiment = 'fmril'
    nor_method = 'indi'
    clf_method = 'svm'
    source_id = 3
    target_id = 2
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

    i = source_id
    j = target_id
    xs = x[indices_sub[i]]
    ys = y_target[indices_sub[i]]
    xt = x[indices_sub[j]]
    yt = y_target[indices_sub[j]]
    note = 'kfold_{}s_{}t_{}_{}_{}_{}_circulaire'.format(i, j, experiment,
                                           nor_method, clf_method, op_function)
    print(note)

    best_params, params_acc = fucs.pairsubs_circular_kfold(xs, ys, xt, yt, ot_method=op_function, clf_method=clf_method)
    reg, eta = best_params
    print('best reg and eta', reg, eta)
    if op_function == 'l1l2':
        transport = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, norm='max')
    elif op_function == 'lpl1':
        transport = ot.da.SinkhornLpl1Transport(reg_e=reg, reg_cl=eta, norm='max')
    elif op_function == 'maplin':
        transport = ot.da.MappingTransport(kernel="linear", mu=reg, eta=eta, bias=True, norm='max')
    else:
        print('Warning: need to choose among "l1l2", "lpl1" and "maplin"', op_function)
    # train on xst, test on xt
    transport.fit(Xs=xs, Xt=xt, ys=ys)
    xst = transport.transform(Xs=xs)
    ot_accs = fucs.get_accuracy(xst, ys, xt, yt, clf_method=clf_method)
    ot_score = np.mean(ot_accs)
    print('train on ot source data, predict on original target, accuracy', ot_score, ot_accs)

    # train on xs, test on xt
    ori_accs = fucs.get_accuracy(xs, ys, xt, yt, clf_method=clf_method)
    ori_score = np.mean(ori_accs)
    print('train on original source data, predict on original target, accuracy', ori_score, ori_accs)
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
    pair_params['params'] = (reg, eta)
    pair_params['params_acc'] = params_acc
    ttest_base_ot = stats.ttest_rel(base_accs,ot_accs)
    pair_params['ttest_base_ot'] = (ttest_base_ot.pvalue, ttest_base_ot.statistic)
    ttest_ori_ot = stats.ttest_rel(ori_accs, ot_accs)
    pair_params['ttest_ori_ot'] = (ttest_ori_ot.pvalue, ttest_ori_ot.statistic)
    print('pair_params', pair_params)
    # if ttest_base_ot .statistic > 0:
    #     joblib.dump(pair_params, result_dir+'/big_base/{}_pair_params.pkl'.format(note))
    # else:
    #     joblib.dump(pair_params, result_dir + '/small_base/{}_pair_params.pkl'.format(note))
    # if ttest_ori_ot .statistic > 0:
    #     joblib.dump(pair_params, result_dir+'/big_ori/{}_pair_params.pkl'.format(note))
    # else:
    #     joblib.dump(pair_params, result_dir + '/small_ori/{}_pair_params.pkl'.format(note))


    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

if __name__ == '__main__':
    main()






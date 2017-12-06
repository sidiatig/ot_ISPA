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
    # op_function = args[5] # 'lpl1' 'l1l2'

    metric_xst = 'null'

    experiment = 'fmril'
    nor_method = 'no'
    clf_method = 'svm'
    source_id = 1
    target_id = 2
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
    x = x_indi.copy() if nor_method == 'indi' else x_data.copy()

    indices_sub = fucs.split_subjects(subjects)
    nb_subs = len(indices_sub)

    i = source_id
    j = target_id
    xs = x[indices_sub[i]]
    ys = y_target[indices_sub[i]]
    xt = x[indices_sub[j]]
    yt = y_target[indices_sub[j]]
    note = '{}s_{}t_{}_{}_{}_sinkhorn_{}_{}'.format(i, j, experiment,
                                                 nor_method, clf_method, op_function,metric_xst)
    print(note)
    reg = 0.01
    eta = 10
    transport = ot.da.SinkhornL1l2Transport
    trans_fuc = transport(reg_e=reg, reg_cl=eta, norm='max')
    trans_fuc.fit(Xs=xs, Xt=xt, ys=ys)
    xst = trans_fuc.transform(Xs=xs)
    print('xst',xst,np.max(xst), np.min(xst),np.mean(xst))
    print('xs',xs,np.max(xs), np.min(xs),np.mean(xs))
    print('xt',xt,np.max(xt), np.min(xt),np.mean(xt))
    xst_accs = fucs.get_accuracy(xst, ys, xt, yt, clf_method=clf_method)
    print(xst_accs)
    y_xs = np.ones(xs.shape[0], dtype=int)
    y_xt = np.zeros(xt.shape[0], dtype=int)
    x = np.vstack((xs, xt))
    print(x.shape)
    y = np.concatenate((y_xs, y_xt))
    print(y.shape)
    # xst has the same labels as xt
    # if acc is low, it means that xst isn't similar to xt.
    # the higher the acc is, the more similar the xst is to xt.
    # xst with the highest acc should be chosen
    xst_accs = fucs.get_accuracy(x, y, xst, y_xt, clf_method=clf_method)
    print('************************************************************************')
    xs_accs1 = fucs.get_accuracy(x, y, xs, y_xs, clf_method=clf_method)
    print('************************************************************************')
    xs_accs2 = fucs.get_accuracy(x, y, xs, y_xt, clf_method=clf_method)
    print('************************************************************************')
    xt_accs = fucs.get_accuracy(x, y, xt, y_xt, clf_method=clf_method)
    print('************************************************************************')
    print('xst_accs',np.mean(xst_accs),xst_accs)
    print('xs_accs1', np.mean(xs_accs1),xs_accs1)
    print('xs_accs2', np.mean(xs_accs2), xs_accs2)
    print('xt_accs', np.mean(xt_accs),xt_accs)


    x_base = x_indi.copy()
    xs_base = x_base[indices_sub[i]]
    ys_base = y_target[indices_sub[i]]
    xt_base = x_base[indices_sub[j]]
    yt_base = y_target[indices_sub[j]]
    base_accs = fucs.get_accuracy(xs_base, ys_base, xt_base, yt_base, clf_method=clf_method)
    print('base_accs',np.mean(base_accs), base_accs)
    xst_accs = fucs.get_accuracy(xst, ys, xt, yt, clf_method=clf_method)
    print('xst_accs', np.mean(xst_accs), xst_accs)
    print('reg,eta', reg, eta)


if __name__ == '__main__':
    main()
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))






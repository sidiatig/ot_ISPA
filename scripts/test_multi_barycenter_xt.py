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


def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'

    args = sys.argv[1:]
    # experiment = args[0]
    # nor_method = args[1]
    # clf_method = args[2]
    # source_id = int(args[3])
    # target_id = int(args[4])
    # op_function = args[5] # 'lpl1' 'l1l2' 'maplin'



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

    # j = target_id
    # xs = [x[indices_sub[i]] for i in source_id]
    # ys = [y_target[indices_sub[i]] for i in source_id]
    # xs = np.vstack(xs)
    # ys = np.concatenate(ys)
    # xt = x[indices_sub[j]]
    # yt = y_target[indices_sub[j]]
    # xsxt = xs.append(xt)

    sub_ids = np.arange(3)
    xsxt = [x_data[indices_sub[i]] for i in sub_ids]
    xsxt_indi = [x_indi[indices_sub[i]] for i in sub_ids]
    ysyt = [y_target[indices_sub[i]] for i in sub_ids]
    trans_xsxt = fucs.inverse_xc_barycenters(xsxt_indi, 'square_loss', 0.1, max_iter=1000, tol=1e-9)
    # trans_xsxt = [StandardScaler().fit_transform(trans_xsxt[i]) for i in sub_ids]

    def split_data(data, i, ids):
        data = np.asarray(data)
        test = data[i]
        train = [data[j] for j in ids if j != i]
        train = np.vstack(train) if len(train[0].shape) > 1 else np.concatenate(train)
        return train, test

    accs = {'trans_accs':[], 'ori_accs':[], 'indi_accs':[]}
    for i in sub_ids:

        trans_xs, trans_xt = split_data(trans_xsxt, i, sub_ids)
        ori_xs, ori_xt = split_data(xsxt, i, sub_ids)
        indi_xs, indi_xt = split_data(xsxt_indi, i, sub_ids)
        ys, yt = split_data(ysyt, i, sub_ids)

        _, _, trans_acc, _ = fucs.gridsearch_persub(trans_xs, ys, trans_xt, yt, clf_method='svm',njobs=3)
        _, _, ori_acc, _ = fucs.gridsearch_persub(ori_xs, ys, ori_xt, yt, clf_method='svm', njobs=3)
        _, _, indi_acc, _ = fucs.gridsearch_persub(indi_xs, ys, indi_xt, yt, clf_method='svm', njobs=3)

        accs['trans_accs'].append((i,trans_acc))
        accs['ori_accs'].append((i, ori_acc))
        accs['indi_accs'].append((i, indi_acc))
        print('xt_id, trans_acc, ori_acc, indi_acc', i, trans_acc, ori_acc, indi_acc)
    print(accs)
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))



if __name__ == '__main__':
    main()






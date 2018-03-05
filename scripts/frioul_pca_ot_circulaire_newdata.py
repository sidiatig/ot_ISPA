# pca + optimal transport + circulaire, several sources one target

# learn d components from source by pca,
# transform them on target
# use circulaire to learn reg, eta for optimal transform
# learn coupling by learned reg eta
# transform target with weighted source

import os
import sys

main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

from data import fmri_localizer as newdata
from model import optimal_classification as ot_fucs

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
import scipy.stats as stats
import numpy as np
import time
import ot


def main():
    main_dir = os.path.split(os.getcwd())[0]
    result_dir = main_dir + '/results'

    args = sys.argv[1:]
    print(args)
    experiment = args[0]
    nor_method = args[1]
    clf_method = args[2]
    ot_method = args[3]
    a = args[4:-1]
    target_subs = int(args[-1])
    a[0] = a[0].replace('[', ' ')
    a[-1] = a[-1].replace(']', ' ')
    for i in range(len(a)):
        a[i] = a[i].replace(',', ' ')
    source_subs = [int(i) for i in a]

    # experiment = 'newdata'
    # nor_method = 'indi'
    # clf_method = 'logis'
    # ot_method = 'l1l2'
    # source_subs = (1,2,3)
    # target_subs = 4

    dim = 100
    note = '{}_{}_{}_{}_{}source_{}target_{}pca_ot_circulaire'.format(experiment,
                                                             nor_method,
                                                             clf_method,
                                                             ot_method,
                                                             source_subs,
                                                             target_subs, dim)
    print(note)
    if experiment == 'newdata':
        source = newdata
    # elif experiment == 'fmrir':
    #     source = fmrir
    # elif experiment == 'meg':
    #     source = meg
    # else:
    #     source = newdata
    result_dir = result_dir + '/pca_ot_circulaire/{}'.format(experiment)
    y = source.y_target
    subjects = np.array(source.subjects)
    ids_subs = source.index_subjects
    x_data = source.x_data_pow if experiment == 'meg' else source.x_data
    x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
    x = x_indi.copy() if nor_method == 'indi' else x_data.copy()

    source_data = [x[ids_subs[i]] for i in source_subs]
    source_data = np.vstack(source_data)
    source_labels = [y[ids_subs[i]] for i in source_subs]
    source_labels = np.hstack(source_labels)
    subs = [subjects[ids_subs[i]] for i in source_subs]
    subs = np.hstack(subs)

    usepca = True
    if usepca:
        pca = PCA(n_components=dim)
        pca.fit(source_data)
        source_pca = pca.transform(source_data)
        print('pca percent, nb of components', np.sum(pca.explained_variance_ratio_),
          len(pca.explained_variance_ratio_))
    clf_data = source_pca if usepca else source_data

    svm_parameters = [{'kernel': ['linear'],
                       'C': [100, 10, 1, 0.1, 0.01, 0.001,
                             0.0001, 0.00001]}]
    logis_parameters = [{'penalty': ['l1', 'l2'],
                         'C': [100, 10, 1, 0.1, 0.01, 0.001,
                               0.0001, 0.00001]}]
    params = logis_parameters if clf_method == 'logis' else svm_parameters

    # clf_scores = []
    # base_scores = []
    j = target_subs
    print('target, {}th sub----------------------------'.format(j))
    x_target = x[ids_subs[j]]
    y_target = y[ids_subs[j]]
    if usepca:
        x_target_pca = pca.transform(x_target)
    pre_data = x_target_pca if usepca else x_target

    #optimal
    reg_eta, params_acc = ot_fucs.pairsubs_circular_kfold(clf_data, source_labels,
                                                          pre_data, ot_method=ot_method,
                                                          clf_method='logis')

    reg, eta = reg_eta
    print('reg, eta, params_acc', reg, eta, params_acc)
    # joblib.dump()

    # reg, eta = 0.001, 10
    # print('reg, eta', reg, eta)
    # acc = ot_fucs.circular_val_kfold(clf_data, labels,
    #                                  pre_data, reg, eta,
    #                                  ot_method='l1l2', clf_method='logis')

    a = np.ones(clf_data.shape[0], dtype=float)/clf_data.shape[0]
    b = np.ones(pre_data.shape[0], dtype=float)/pre_data.shape[0]
    M = ot.utils.dist(clf_data, pre_data)
    M /= M.max()

    gamma = ot.da.sinkhorn_l1l2_gl(a, source_labels, b, M, reg, eta=eta, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False)
    gamma = np.transpose(gamma)
    transp = gamma/ np.sum(gamma, 1)[:, None]
    # set nans to 0
    transp[~ np.isfinite(transp)] = 0
    # compute transported samples
    data_source = clf_data
    data_target = np.dot(transp, data_source)

    # logo = LeaveOneGroupOut()
    # grid_cv = list(logo.split(data_source, labels, subs))
    clf = LogisticRegression if clf_method == 'logis' else SVC
    skf = StratifiedKFold(n_splits=10)
    grid_cv = list(skf.split(data_source, source_labels))
    grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=3)
    grid_clf.fit(data_source, source_labels)
    print('grid_clf params', grid_clf.best_params_)
    target_pre = grid_clf.predict(data_target)
    print('clf pre', target_pre)
    clf_score = accuracy_score(y_target, target_pre)
    print('clf score', clf_score)
    # clf_scores.append(clf_score)

    # base prediction
    base_clf = LogisticRegression if clf_method == 'logis' else SVC
    logo = LeaveOneGroupOut()
    base_cv = list(logo.split(source_data, source_labels, subs))
    # skf = StratifiedKFold(n_splits=10)
    # base_cv = list(skf.split(source_data, source_labels))
    base_gridclf = GridSearchCV(base_clf(), params, cv=base_cv, n_jobs=3)
    base_gridclf.fit(source_data, source_labels)
    print('base_gridclf params', base_gridclf.best_params_)

    # base_gridclf = LogisticRegression(penalty='l2', C=0.0001)
    # base_gridclf.fit(source_data, source_labels)
    base_pre = base_gridclf.predict(x_target)
    print('base_pre', base_pre)
    base_score = accuracy_score(y_target, base_pre)
    print('base score', base_score)
    # base_scores.append(base_score)

    # print('base_scores', np.mean(base_scores), base_scores)
    # print('clf_scores', np.mean(clf_scores), clf_scores)
    # stats_resutls = stats.ttest_rel(base_scores, clf_scores)
    # print(stats_resutls)
    result = {}
    result['reg_eta'] = (reg, eta)
    result['params_acc'] = params_acc
    result['params_clf'] = grid_clf.best_params_
    result['params_base'] = base_gridclf.best_params_
    result['pca_d'] = dim
    result['base_pre'] = base_pre
    result['target_pre'] = target_pre
    result['clf_base_acc'] = (clf_score, base_score)

    joblib.dump(result, result_dir + '/{}.pkl'.format(note))
    print(note)

if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    main()
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))





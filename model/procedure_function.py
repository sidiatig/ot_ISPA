import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
import scipy.stats as stats
import pandas as pd
import ot

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, \
    GroupKFold, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

def shuffle_data(x,y,nb_persub):
    # shuffle each subject's data
    # x is n subjects' data
    x_shuffle = np.empty(x.shape)
    y_shuffle = np.empty(y.shape)
    ids_shuffle = []
    for i in range(int(x.shape[0] / nb_persub)):
        start = i * nb_persub
        end = (i + 1) * nb_persub
        index_shuffle = [j for j in range(start, end)]
        np.random.shuffle(index_shuffle)
        ids_shuffle.append(index_shuffle)
        x_shuffle[start:end] = x[index_shuffle]
        y_shuffle[start:end] = y[index_shuffle]

    # np.save(result_dir + 'datasave/shuffle_ids_{}_{}_{}_{}r_{}subs_{}trainsize_{}totalrounds.npy'.
    #         format(experiment, clf_method, data_method, r, k, trainsize, nrounds),
    #         ids_shuffle)

    return x_shuffle, y_shuffle

def split_subjects(subjects):
    # separate subjects
    index_subjects = []
    for subject in np.unique(subjects):
        index = [j for j, s in enumerate(subjects) if s == subject]
        index_subjects.append(index)
    return index_subjects

def gridsearch(x, y_target, subjects, cross_v, experiment, clf_method,
               nor_method, cv_start, njobs=1):

    note = '{}_{}_{}_{}cvstart'.format(experiment, nor_method, clf_method,
                                          cv_start)
    svm_parameters = [{'kernel': ['linear'],
                       'C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]

    logis_parameters = [{'penalty': ['l1', 'l2'],
                         'C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
    clf = LogisticRegression if clf_method == 'logis' else SVC
    params = logis_parameters if clf_method == 'logis' else svm_parameters

    train, test = cross_v
    x_train = x[train]
    y_train = y_target[train]
    x_test = x[test]
    y_test = y_target[test]
    subjects_train = subjects[train]
    if experiment == 'meg':
        # gkf = GroupKFold(n_splits=7)
        # grid_cv = list(gkf.split(x_train, y_train, subjects_train))
        logo = LeaveOneGroupOut()
        grid_cv = list(logo.split(x_train, y_train, subjects_train))

    else:
        # leave 2 subjects out for fmri in gridsearch
        gkf = GroupKFold(n_splits=45)
        grid_cv = list(gkf.split(x_train, y_train, subjects_train))

        # logo = LeaveOneGroupOut()
        # grid_cv = list(logo.split(x_train, y_train, subjects_train))
        # print('logo')
    grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=njobs)
    grid_clf.fit(x_train, y_train)
    print('best params', grid_clf.best_params_)
    joblib.dump(grid_clf.best_params_,
                result_dir + '/gridtables/{}cv_gridbest.pkl'.format(note))
    # grid_bestpara = joblib.load(result_dir + 'joblib.pkl')
    grid_csv = pd.DataFrame.from_dict(grid_clf.cv_results_)
    with open(result_dir + '/gridtables/{}cv_gridtable.csv'.format(note), 'w') as f:
        grid_csv.to_csv(f)

    pre = grid_clf.predict(x_test)
    score_op = accuracy_score(y_test, pre)
    print('optimal transport accuracy of {}th split:'.format(cv_start), score_op)
    return score_op

def gridsearch_persub(xs, ys, xt, yt, clf_method='logis',njobs=3):
    if clf_method != 'knn':
        svm_parameters = [{'kernel': ['linear'],
                           'C': [100, 10, 1, 0.1, 0.01, 0.001,
                                   0.0001, 0.00001]}]
        logis_parameters = [{'penalty': ['l1', 'l2'],
                             'C': [100, 10, 1, 0.1, 0.01, 0.001,
                                   0.0001, 0.00001]}]
        clf = LogisticRegression if clf_method == 'logis' else SVC
        params = logis_parameters if clf_method == 'logis' else svm_parameters

        skf = StratifiedKFold(n_splits=10)
        grid_cv = list(skf.split(xs, ys))
        grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=njobs)
        grid_clf.fit(xs, ys)
        grid_best = grid_clf.best_params_
        grid_csv = pd.DataFrame.from_dict(grid_clf.cv_results_)
        pre = grid_clf.predict(xt)
        score_op = accuracy_score(yt, pre)
        return grid_best, grid_csv, score_op, pre
    else:
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(xs, ys)
        pre = neigh.predict(xt)
        score_op = accuracy_score(yt, pre)
        return 0, 0, score_op, pre


def pairsubs_sinkhorn_lables_h(xs, ys, xt, ot_method='l1l2', clf_method='logis'):
    # metric:  ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
    # ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’,
    # ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
    # ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.

    # use different sets of reg and eta to transport source,
    # find the best transport source based on hausdorff distance
    # reg can't be too small, otherwise K will be 0
    # when reg is larger than 1, xst is too compacted
    regs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    etas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    # regs = [1e-2, 1e-1]
    # etas = [ 1e-1]
    if ot_method == 'l1l2':
        transport = ot.da.SinkhornL1l2Transport
    elif ot_method == 'lpl1':
        transport = ot.da.SinkhornLpl1Transport
    else:
        print('Warning: need to choose between "l1l2" and "lpl1" ', ot_method)
    params_acc= {'params':[], 'acc':[] }
    best_dist = -1
    for reg in regs:
        for eta in etas:
            print('reg:{}, eta:{}'.format(reg,eta))
            trans_fuc = transport(reg_e=reg, reg_cl=eta, norm='max')
            trans_fuc.fit(Xs=xs, Xt=xt, ys=ys)
            xst = trans_fuc.transform(Xs=xs)
            accs = h_divergence(xst, xt, clf_method=clf_method)
            dist = np.mean(accs)
            params_acc['params'].append([reg, eta])
            params_acc['acc'].append(dist)
            print('reg,eta, h dist', reg, eta, dist)
            if best_dist == -1:
                if dist >= 0:
                    best_dist = dist
                    best_accs = accs
                    best_params = (reg,eta)
                    best_xst = xst
            elif dist >= 0:
                ttest = stats.ttest_rel(best_accs, accs)
                if ttest.statistic > 0:
                    best_accs = accs
                    best_params = (reg, eta)
                    best_xst = xst
    params_acc['params'] = np.asarray(params_acc['params'])
    params_acc['acc'] = np.asarray(params_acc['acc'])
    print('best reg and eta', best_params)
    return best_xst, best_params, params_acc

def h_divergence(xst, xt, clf_method='logis'):
    # train a classifier to distinguish xst and xt,
    # classifier is trained on each pair of xst and xt,
    # take xst and xt as two classes within one subject,
    # use multi splits within subject to get a set of accuracies,
    # t test between different sets of accuracies,
    # the average acc of a set should be close to 0.5
    y_xst = np.ones(xst.shape[0], dtype=int)
    y_xt = np.zeros(xt.shape[0],dtype=int)
    x = np.vstack((xst,xt))
    y = np.concatenate((y_xst,y_xt))
    accs = []
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    for train, test in rskf.split(x, y):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        grid_best, grid_csv, acc, _ = gridsearch_persub(x_train, y_train,
                                                     x_test, y_test,
                                                     clf_method=clf_method)
        # print('gridsearch for h, best classifier params', grid_best)
        # print(grid_csv)
        accs.append(acc-0.5)
    return accs

def get_accuracy(xs, ys, xt, yt, n_spl=10, n_rep=1, clf_method='logis'):
    # train a clf on part of (xs,ys), get acc on xt,
    # do it for several times, average all the accs to get the final acc and return
    # if the train data is small, must use more splits to make the results more rebust
    accs = []
    rskf = RepeatedStratifiedKFold(n_splits=n_spl, n_repeats=n_rep)
    for train, test in rskf.split(xs,ys):
        x_train = xs[train]
        y_train = ys[train]
        grid_best, grid_csv, acc, _ = gridsearch_persub(x_train,y_train,xt,yt,clf_method=clf_method)
        # print('gridsearch best params', grid_best)
        # print(grid_csv)
        accs.append(acc)
    return accs

def acc_circular(x1, y1, x2, y2, xt, yt, reg, eta, ot_method='maplin', clf_method='logis'):
    if ot_method == 'l1l2':
        transport = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, norm='max')
    elif ot_method == 'lpl1':
        transport = ot.da.SinkhornLpl1Transport(reg_e=reg, reg_cl=eta, norm='max')
    elif ot_method == 'maplin':
        transport = ot.da.MappingTransport(kernel="linear", mu=reg, eta=eta, bias=True, norm='max')
    else:
        print('Warning: need to choose among "l1l2", "lpl1" and "maplin"', ot_method)
    transport.fit(Xs=x1, Xt=xt, ys=y1)
    xst1 = transport.transform(Xs=x1)
    xst2 = transport.transform(Xs=x2)
    _, _, _, yt1 = gridsearch_persub(xst1, y1, xt, yt, clf_method=clf_method)
    acc = np.mean(get_accuracy(xt, yt1, xst2, y2, clf_method=clf_method))
    return acc

def circular_val(xs, ys, xt, yt, reg, eta, ot_method='maplin', clf_method='logis'):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    s1s = []
    s2s = []
    for train_index, test_index in sss.split(xs, ys):
        xs1, xs2 = xs[train_index], xs[test_index]
        ys1, ys2 = ys[train_index], ys[test_index]
        s1 = acc_circular(xs1, ys1, xs2, ys2, xt, yt, reg, eta, ot_method=ot_method, clf_method=clf_method)
        s2 = acc_circular(xs2, ys2, xs1, ys1, xt, yt, reg, eta, ot_method=ot_method, clf_method=clf_method)
        s1s.append(s1)
        s2s.append(s2)
    s1_mean = np.mean(s1s)
    s2_mean = np.mean(s2s)
    print('s1,s2', s1_mean, s2_mean)
    return (s1_mean + s2_mean)/2


def pairsubs_circular(xs, ys, xt, yt, ot_method='maplin', clf_method='logis'):

    regs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    etas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    # regs = [1e-3]
    # etas = [1, 10, 100]
    best_acc = 0
    for reg in regs:
        for eta in etas:
            print('reg:{}, eta:{}'.format(reg, eta))
            s = circular_val(xs, ys, xt, yt, reg, eta, ot_method=ot_method, clf_method=clf_method)
            print('reg,eta, circular acc', reg, eta, s)
            if best_acc == 0:
                best_acc = s
                best_params = (reg, eta)
            else:
                if s > best_acc:
                    best_acc = s
                    best_params = (reg, eta)

    return best_params

# def acc_circular_kfold(x1, y1, x2, y2, xt, yt, reg, eta, ot_method='maplin', clf_method='logis'):
#     if ot_method == 'l1l2':
#         transport = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, norm='max')
#     elif ot_method == 'lpl1':
#         transport = ot.da.SinkhornLpl1Transport(reg_e=reg, reg_cl=eta, norm='max')
#     elif ot_method == 'maplin':
#         transport = ot.da.MappingTransport(kernel="linear", mu=reg, eta=eta, bias=True, norm='max')
#     else:
#         print('Warning: need to choose among "l1l2", "lpl1" and "maplin"', ot_method)
#     transport.fit(Xs=x1, Xt=xt, ys=y1)
#     xst1 = transport.transform(Xs=x1)
#     xst2 = transport.transform(Xs=x2)
#     _, _, _, yt1 = gridsearch_persub(xst1, y1, xt, yt, clf_method=clf_method)
#     _, _, acc, _ = gridsearch_persub(xt, yt1, xst2, y2, clf_method=clf_method)
#     return acc

def circular_val_kfold(xs, ys, xt, yt, reg, eta, ot_method='maplin', clf_method='logis'):
    if ot_method=='maplin':
        skf = StratifiedKFold(n_splits=10)
        ss = []
        for train_index, test_index in skf.split(xs, ys):
            xs1, xs2 = xs[train_index], xs[test_index]
            ys1, ys2 = ys[train_index], ys[test_index]
            acc = acc_circular(xs1, ys1, xs2, ys2, xt, yt, reg, eta, ot_method=ot_method, clf_method=clf_method)
            ss.append(acc)
        s_mean = np.mean(ss)
        print('s_mean', s_mean)
        return s_mean
    else:
        if ot_method == 'l1l2':
            transport = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, norm='max')
        elif ot_method == 'lpl1':
            transport = ot.da.SinkhornLpl1Transport(reg_e=reg, reg_cl=eta, norm='max')
        elif ot_method == 'maplin':
            transport = ot.da.MappingTransport(kernel="linear", mu=reg, eta=eta, bias=True, norm='max')
        else:
            print('Warning: need to choose among "l1l2", "lpl1" and "maplin"', ot_method)
        transport.fit(Xs=x1, Xt=xt, ys=y1)
        xst1 = transport.transform(Xs=x1)
        xst2 = transport.transform(Xs=x2)
        _, _, _, yt1 = gridsearch_persub(xst1, y1, xt, yt, clf_method=clf_method)
        acc = np.mean(get_accuracy(xt, yt1, xst2, y2, clf_method=clf_method))


def pairsubs_circular_kfold(xs, ys, xt, yt, ot_method='maplin', clf_method='logis'):
    # separate xs into to 2 parts based on 10 kfolds,
    # use the first part xs1 and xt to learn xst1,
    # use xst1 to learn a clf then predict on xt, get the predictions yt1
    # use xt, yt1 to learn 2-nd clf, use it to predict on xst2, get acc of xst2
    # use mean acc from those 10 kfolds to choose best reg and eta
    regs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    etas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    # regs = [1e-3]
    # etas = [ 1, 10, 100]
    best_acc = 0
    for reg in regs:
        for eta in etas:
            print('reg:{}, eta:{}'.format(reg, eta))
            s = circular_val_kfold(xs, ys, xt, yt, reg, eta, ot_method=ot_method, clf_method=clf_method)
            print('reg,eta, circular kfold acc', reg, eta, s)
            if best_acc == 0:
                best_acc = s
                best_params = (reg, eta)
            else:
                if s > best_acc:
                    best_acc = s
                    best_params = (reg, eta)
    return best_params





        # def pairsubs_sinkhorn(xs, xt, metric="hausdorff"):
#     # use different sets of reg to transport source with sinkhorn,
#     # find the best transport source based on hausdorff distance
#     # reg can't be too small, otherwise K will be 0
#
#     regs = [1e-3, 1e-2, 1e-1, 1, 10]
#     best_dist = 0
#
#     transport = ot.da.SinkhornTransport
#     for reg in regs:
#         print('reg:{}'.format(reg))
#         trans_fuc = transport(reg_e=reg, norm='max')
#         trans_fuc.fit(Xs=xs, Xt=xt)
#         xst = trans_fuc.transform(Xs=xs)
#         if metric=='hausdorff':
#             dic1 = directed_hausdorff(xst, xt)[0]
#             dic2 = directed_hausdorff(xt, xst)[0]
#             dist = max(dic1, dic2)
#         else:
#             dist = np.sum(cdist(xst,xt, metric=metric))
#         if best_dist == 0:
#             best_dist = dist
#             best_params = reg
#             best_xst = xst
#         if dist < best_dist:
#             best_dist = dist
#             best_params = reg
#             best_xst = xst
#     return best_xst, best_params
#
#





#
# def paird_ot_transport(xs, ys, xt, yt, experiment='fmril', op_function='sinkhorn' ):
#
#     M = ot.utils.dist(xs, xt)
#     M = ot.utils.cost_normalization(M, 'max')
#     a = np.ones(xs.shape[0]) / xs.shape[0]
#     b = np.ones(xt.shape[0]) / xt.shape[0]
#
#     ori_best, ori_grid, ori_score = gridsearch_persub(xs, ys, xt, yt)
#     ot_scores = []
#     ot_params = {}
#     ot_params['ori_score'] = ori_score
#     # reg can't be too small, otherwize K will be 0
#     # when reg is larger , prediction score is better,
#     for reg in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
#         if op_function=='sinkhorn':
#             G, cost_dist = opt.bregman.sinkhorn_knopp(a=a,b=b,M=M, reg=reg)
#             xst = opt.da.transform(G, xt)
#             best_param, grid_cvs, score_ot = gridsearch_persub(xst, ys, xt, yt, experiment=experiment)
#             if len(ot_scores) == 0:
#                 ot_scores.append(score_ot)
#                 ot_params['reg'] = reg
#                 ot_params['C'] = best_param['C']
#                 ot_params['penalty'] = best_param['penalty']
#                 ot_params['score_ot'] = score_ot
#                 ot_params['op_function'] = op_function
#                 ot_params['xst'] = xst
#                 ot_params['cost_dist'] = cost_dist
#             else:
#                 if score_ot > max(ot_scores):
#                     ot_params['reg'] = reg
#                     ot_params['C'] = best_param['C']
#                     ot_params['penalty'] = best_param['penalty']
#                     ot_params['score_ot'] = score_ot
#                     ot_params['xst'] = xst
#                     ot_params['cost_dist'] = cost_dist
#                 ot_scores.append(score_ot)
#         else:
#             for eta in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:
#                 if op_function == 'lpl1':
#                     G, cost_dist = ot.da.sinkhorn_lpl1_mm(a=a, labels_a=ys, b=b, M=M, reg=reg, eta=eta)
#                     xst = opt.da.transform(G, xt)
#                 elif op_function == 'l1l2':
#                     G, cost_dist = opt.da.sinkhorn_l1l2_gl(a=a, labels_a=ys, b=b, M=M, reg=reg, eta=eta)
#                     xst = opt.da.transform(G, xt)
#                 best_param, grid_cvs, score_ot = gridsearch_persub(xst, ys, xt, yt, experiment=experiment)
#                 if len(ot_scores) == 0:
#                     ot_scores.append(score_ot)
#                     ot_params['reg'] = reg
#                     ot_params['eta'] = eta
#                     ot_params['C'] = best_param['C']
#                     ot_params['penalty'] = best_param['penalty']
#                     ot_params['score_ot'] = score_ot
#                     ot_params['op_function'] = op_function
#                     ot_params['xst'] = xst
#                     ot_params['cost_dist'] = cost_dist
#                 else:
#                     if score_ot > max(ot_scores):
#                         ot_params['reg'] = reg
#                         ot_params['eta'] = eta
#                         ot_params['C'] = best_param['C']
#                         ot_params['penalty'] = best_param['penalty']
#                         ot_params['score_ot'] = score_ot
#                         ot_params['xst'] = xst
#                         ot_params['cost_dist'] = cost_dist
#                     ot_scores.append(score_ot)
#     return ot_params
#
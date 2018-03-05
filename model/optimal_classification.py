import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

import numpy as np
import pandas as pd
import ot

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, \
    GroupKFold, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit


def getAdaptedData(sourceData, targetData):
    d = 40
    XS = np.transpose(PCA(d).fit(sourceData).components_)
    XT = np.transpose(PCA(d).fit(targetData).components_)
    Xa = XS.dot(np.transpose(XS)).dot(XT)

    Sa = sourceData.dot(Xa)
    Tt = targetData.dot(XT)
    return (Sa, Tt)

def gridsearch_persub(xs, ys, xt, clf_method='logis',njobs=3):

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
    return grid_best, grid_csv, pre

def get_accuracy(xs, ys, xt, yt, clf_method='logis'):
    # train a clf on part of (xs,ys), get prediction on xt,
    # do it for several times, average all the accs to get the final acc and return
    # if the train data is small, must use more splits to make the results more rebust
    accs = []
    skf = StratifiedKFold(n_splits=10)
    for train, test in skf.split(xs,ys):
        x_train = xs[train]
        y_train = ys[train]
        grid_best, grid_csv, pre = gridsearch_persub(x_train, y_train, xt, clf_method=clf_method)
        acc = accuracy_score(yt, pre)
        # print('gridsearch best params', grid_best)
        # print(grid_csv)
        accs.append(acc)
    return accs

def cir_val(xs1, ys1, xs2, ys2, xt_hat, clf_method='logis'):
    _, _, yt_hat = gridsearch_persub(xs1, ys1, xt_hat, clf_method=clf_method)

    acc = np.mean(get_accuracy(xt_hat, yt_hat, xs2, ys2, clf_method=clf_method))
    return acc


def circular_val_kfold(xs, ys, xt, reg, eta, ot_method='l1l2', clf_method='logis'):
    # maplin: split xs into two parts, learn mapping from the first part
    # l1l2 and l1lp: first learn mapping from xs and xt, then split xst into two parts
    # 10 folds to get average acc under parameters of reg and eta
    skf = StratifiedKFold(n_splits=10)
    ss = []
    if ot_method == 'maplin':
        transport = ot.da.MappingTransport(kernel="linear", mu=reg, eta=eta, bias=True, norm='max')
        for train_index, test_index in skf.split(xs, ys):
            xs1, xs2 = xs[train_index], xs[test_index]
            ys1, ys2 = ys[train_index], ys[test_index]
            transport.fit(Xs=xs1, Xt=xt)
            xst1 = transport.transform(Xs=xs1)
            xst2 = transport.transform(Xs=xs2)
            acc = cir_val(xst1, ys1, xst2, ys2, xt, clf_method=clf_method)
            ss.append(acc)
    else:
        if ot_method == 'l1l2':
            transport = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, norm='max')
        elif ot_method == 'lpl1':
            transport = ot.da.SinkhornLpl1Transport(reg_e=reg, reg_cl=eta, norm='max')
        else:
            print('Warning: need to choose among "l1l2", "lpl1"', ot_method)
        transport.fit(Xs=xs, Xt=xt, ys=ys)
        gamma = np.transpose(transport.coupling_)
        transp = gamma / np.sum(gamma, 1)[:, None]
        # set nans to 0
        transp[~ np.isfinite(transp)] = 0
        # compute transported samples
        xt_hat = np.dot(transp, xs)
        xt_hat = StandardScaler().fit_transform(xt_hat)
        for train_index, test_index in skf.split(xs, ys):
            xs1, xs2 = xs[train_index], xs[test_index]
            ys1, ys2 = ys[train_index], ys[test_index]
            acc = cir_val(xs1, ys1, xs2, ys2, xt_hat, clf_method=clf_method)
            ss.append(acc)
    s_mean = np.mean(ss)
    return s_mean

def pairsubs_circular_kfold(xs, ys, xt, ot_method='l1l2', clf_method='logis'):
    # separate xs into to 2 parts based on 10 kfolds,
    # use the first part xs1 and xt to learn xst1,
    # use xst1 to learn a clf then predict on xt, get the predictions yt1
    # use xt, yt1 to learn 2-nd clf, use it to predict on xst2, get acc of xst2
    # use mean acc from those 10 kfolds to choose best reg and eta
    regs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    etas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    # regs = [1e-3]
    # etas = [10]
    params_acc = {'params': [], 'acc': []}
    best_acc = 0
    for reg in regs:
        for eta in etas:
            try:
                s = circular_val_kfold(xs, ys, xt, reg, eta, ot_method=ot_method, clf_method=clf_method)
            except:
                # print('reg:{}, eta:{}, except error'.format(reg, eta))
                s = 0
            print('reg,eta, circular kfold acc', reg, eta, s)
            params_acc['params'].append([reg, eta])
            params_acc['acc'].append(s)
            if best_acc == 0:
                best_acc = s
                best_params = (reg, eta)
            else:
                if s > best_acc:
                    best_acc = s
                    best_params = (reg, eta)
    params_acc['params'] = np.asarray(params_acc['params'])
    params_acc['acc'] = np.asarray(params_acc['acc'])
    return best_params, params_acc

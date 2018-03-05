import os
import sys
import matplotlib
matplotlib.use('Agg')

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from model import procedure_function as fucs
from model import dual_optima_barycenter as dual
from model import optimal_classification as ot_fucs

from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import itertools
import numpy as np
import time
import ot

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

experiment = 'fmril'
nor_method = 'indi'
dist_type = 'matrix'
reg = 1/1000
# ids = [i for i in range(10)]
ite = 500
# pos_neg = 'neg'

if experiment == 'fmril':
    source = fmril


result_dir = result_dir + '/barycenter_clf/{}'.format(experiment)
y_target = source.y_target
subjects = np.array(source.subjects)
ids_subs = source.index_subjects
x_data = source.x_data_pow if experiment == 'meg' else source.x_data
x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
x = x_indi.copy() if nor_method == 'indi' else x_data.copy()
#
# barys_pos = []
# barys_neg = []
onesub = [i for i in range(20)]
# twosubs = list(itertools.combinations([i for i in range(20)],2))
# threesubs = list(itertools.combinations([i for i in range(20)],3))[:400]
# for nsubs in [threesubs, onesub, twosubs]:
#     for id in nsubs:
#         data_pos = np.load(result_dir +'/{}_{}_sub{}_{}_{}reg_{}ite_pos.npz'.format(experiment,
#                                                                                nor_method, id,
#                                                                                dist_type, reg, ite))
#         data_neg = np.load(result_dir +'/{}_{}_sub{}_{}_{}reg_{}ite_neg.npz'.format(experiment,
#                                                                                nor_method, id,
#                                                                                dist_type, reg, ite))
#         barys_neg.append(data_neg['bary_wass'])
#         barys_pos.append(data_pos['bary_wass'])
#
# barys = np.vstack((np.vstack(barys_pos), np.vstack(barys_neg)))
# print('barys shape', barys.shape)
# # barys_norm = StandardScaler().fit_transform(barys)
# labels = np.hstack((np.ones(len(barys_pos)), np.zeros(len(barys_neg))))

barys = x[:800]
labels = y_target[:800]
subs = subjects[:800]
pca = PCA(n_components=40)
pca.fit(barys)
print('pca percent', np.sum(pca.explained_variance_ratio_))
barys_pca = pca.transform(barys)

usepca = True
print('usepca', usepca)
clf_data = barys_pca if usepca else barys

# knn = KNeighborsClassifier(n_neighbors=5)

svm_parameters = [{'kernel': ['linear'],
                   'C': [100, 10, 1, 0.1, 0.01, 0.001,
                         0.0001, 0.00001]}]
logis_parameters = [{'penalty': ['l1', 'l2'],
                     'C': [100, 10, 1, 0.1, 0.01, 0.001,
                           0.0001, 0.00001]}]
clf_method = 'logis'
clf = LogisticRegression if clf_method == 'logis' else SVC
params = logis_parameters if clf_method == 'logis' else svm_parameters

# grid_cv = list(skf.split(clf_data, labels))
# grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=3)
# grid_clf.fit(clf_data, labels)
# print('grid_clf params', grid_clf.best_params_)

# grid_clf = LogisticRegression(penalty='l2', C=100)
# grid_clf.fit(barys, labels)

x_train = [x[ids_subs[i]] for i in onesub]
x_train = np.vstack(x_train)
y_train = [y_target[ids_subs[i]] for i in onesub]
y_train = np.hstack(y_train)
subs_train = [subjects[ids_subs[i]] for i in onesub]
subs_train = np.hstack(subs_train)

# base_clf = LogisticRegression if clf_method == 'logis' else SVC
# logo = LeaveOneGroupOut()
# base_cv = list(logo.split(x_train, y_train, subs_train))
# base_gridclf = GridSearchCV(base_clf(), params, cv=base_cv, n_jobs=3)
# base_gridclf.fit(x_train, y_train)
# print('base_gridclf params', base_gridclf.best_params_)

base_gridclf = LogisticRegression(penalty='l2', C=0.0001)
base_gridclf.fit(x_train, y_train)

clf_scores = []
base_scores = []
for j in range(21,100):
    print('jth target----------------------------', j)
    x_target = x[ids_subs[j]]
    x_target_pca = pca.transform(x_target)
    y_label = y_target[ids_subs[j]]
    pre_data = x_target_pca if usepca else x_target
    #subspace
    # data_source, data_target = getAdaptedData(clf_data, pre_data)
    # print('subspace')
    data_source, data_target = clf_data, pre_data
    # print('without subspace')

    #optimal

    reg_eta, params_acc = ot_fucs.pairsubs_circular_kfold(data_source, labels,
                                                          data_target,
                                                           ot_method='l1l2', clf_method='logis')
    reg, eta = reg_eta
    print('reg, eta, params_acc', reg, eta, params_acc)

    # reg, eta = 0.001, 10
    # print('reg, eta', reg, eta)
    # acc = ot_fucs.circular_val_kfold(clf_data, labels, pre_data, reg, eta, ot_method='l1l2', clf_method='logis')



    a = np.ones(clf_data.shape[0], dtype=float)/clf_data.shape[0]
    b = np.ones(pre_data.shape[0], dtype=float)/pre_data.shape[0]
    M = ot.utils.dist(clf_data, pre_data)
    M /= M.max()

    gamma = ot.da.sinkhorn_l1l2_gl(a, labels, b, M, reg, eta=eta, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False)
    gamma = np.transpose(gamma)
    data_source = clf_data
    transp = gamma/ np.sum(gamma, 1)[:, None]
    # set nans to 0
    transp[~ np.isfinite(transp)] = 0
    # compute transported samples
    data_target = np.dot(transp, data_source)

    # logo = LeaveOneGroupOut()
    # grid_cv = list(logo.split(data_source, labels, subs))

    skf = StratifiedKFold(n_splits=10)
    grid_cv = list(skf.split(data_source, labels))
    grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=3)
    grid_clf.fit(data_source, labels)
    print('grid_clf params', grid_clf.best_params_)

    clf_pre = grid_clf.predict(data_target)
    print('clf pre', clf_pre)
    clf_score = accuracy_score(y_label, clf_pre)
    print('clf score', clf_score)
    clf_scores.append(clf_score)

    # base prediction
    base_pre = base_gridclf.predict(x_target)
    # print('base pre', base_pre)
    base_score = accuracy_score(y_label, base_pre)
    print('base score', base_score)
    base_scores.append(base_score)
print('base_scores', np.mean(base_scores), base_scores)
print('clf_scores', np.mean(clf_scores), clf_scores)
stats_resutls = stats.ttest_rel(base_scores, clf_scores)
print(stats_resutls)



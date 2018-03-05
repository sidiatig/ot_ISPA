import numpy as np
from sklearn.model_selection import LeavePGroupsOut, GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import pandas as pd
import os


main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

def split_subjects(subjects):
    # separate subjects
    index_subjects = []
    for subject in np.unique(subjects):
        index = [j for j, s in enumerate(subjects) if s == subject]
        index_subjects.append(index)
    return index_subjects


def create_cv(x, y, subjects, P):
    """

    :param x:
    :param y:
    :param N:
    :return:
    """
    cv = []
    lpgo = LeavePGroupsOut(n_groups=P)
    for train_index, test_index in lpgo.split(x, y, subjects):
        cv.append((train_index, test_index))
    return cv

def n2n_gridsearch(x_train, y_train, x_test, y_test, subjects_train, clf_method, K, njobs=3, grid=False):

    svm_parameters = [{'kernel': ['linear'],
                       'C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]

    logis_parameters = [{'penalty': ['l1', 'l2'],
                         'C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
    clf = LogisticRegression if clf_method == 'logis' else SVC
    params = logis_parameters if clf_method == 'logis' else svm_parameters

    gkf = GroupKFold(n_splits=K)
    grid_cv = list(gkf.split(x_train, y_train, subjects_train))
    grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=njobs)
    grid_clf.fit(x_train, y_train)
    best_params = grid_clf.best_params_

    # joblib.dump(grid_clf.best_params_,
    #             result_dir + '/gridtables/{}cv_gridbest.pkl'.format(note))
    # grid_bestpara = joblib.load(result_dir + 'joblib.pkl')
    grid_csv = pd.DataFrame.from_dict(grid_clf.cv_results_)
    # with open(result_dir + '/gridtables/{}cv_gridtable.csv'.format(note), 'w') as f:
    #     grid_csv.to_csv(f)

    pre = grid_clf.predict(x_test)
    score_op = accuracy_score(y_test, pre)

    if grid:
        return score_op, best_params, pre, grid_csv
    else:
        return score_op, best_params, pre




# def one2one():
#     return
#
#
# def n2n()



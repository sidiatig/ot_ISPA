import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, \
    GroupKFold, StratifiedKFold
from sklearn.externals import joblib

import fmri_data_cv as fmril
import fmri_data_cv_rh as fmrir
import meg_data_cv as meg

import time
import sys
import itertools

result_dir = '/hpc/crise/wang.q/results/ISPA/experiment2/meg_same_size/'

def shuffle_data(x,y,nb_persub):
    # shuffle each subject's data
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
    ids_shuffle = np.vstack(ids_shuffle)
    # np.save(result_dir + 'datasave/shuffle_ids_{}_{}_{}_{}r_{}subs_{}trainsize_{}totalrounds.npy'.
    #         format(experiment, clf_method, data_method, r, k, trainsize, nrounds),
    #         ids_shuffle)

    return x_shuffle, y_shuffle,ids_shuffle

def split_subjects(subjects):
    index_subjects = []

    for subject in np.unique(subjects):
        index = [j for j, s in enumerate(subjects) if s == subject]
        index_subjects.append(index)
    return index_subjects

def split_Ds_Dm(x,y,subjects,nb_persub, n=2):
    # n is the number of subjects in Dm, for meg, n=2,for fmri, n=50
    # split x into Ds and Dm
    index_subjects = split_subjects(subjects)

    # Ds_Dm_split = list(itertools.combinations([i for i in range(int(x.shape[0]/nb_persub))],2))
    # np.random.shuffle(Ds_Dm_split)
    # np.save(result_dir+'datasave/Ds_Dm_split_ids_{}_{}_{}_{}r_{}subs_{}trainsize_{}totalrounds_10splits.npy'.
    #         format(experiment, clf_method, data_method, r, k, trainsize, nrounds),
    #         Ds_Dm_split[:2])

    # 14 subjects for Ds, 2 subjects for Dm
    Ds_Dm_split = np.random.permutation(int(x.shape[0]/nb_persub))[:n]
    Dm_id = []
    for j in Ds_Dm_split:
        for i in index_subjects[j]:
            Dm_id.append(i)
    Dm_id = np.array(Dm_id)
    Ds_id = np.delete(np.arange(x.shape[0]), Dm_id)
    # set Ds and Dm
    Ds = x[Ds_id]
    Ys = y[Ds_id]
    Dm = x[Dm_id]
    Ym = y[Dm_id]
    return  Ds, Ys, Dm, Ym, Ds_Dm_split

def ids_train_test(Ds, Ys, nb_persub, k, r, subjects):
    # randomly choose k subs from Ds,
    index_subjects = split_subjects(subjects)
    nb_subs = int(Ds.shape[0] / nb_persub)
    choose_sub = np.random.permutation(nb_subs)[:k]
    # f-fold cross validation within each subject
    f = int(1 / (1 - r))
    # train and test index of each chosen subject's data in each fold
    TRid_ffold = [[] for j in range(f)]
    Tsid_ffold = [[] for j in range(f)]
    trainsubs_ffold = []
    for s in choose_sub:
        data_s = Ds[index_subjects[s]]
        label_s = Ys[index_subjects[s]]
        count_ffold = 0
        skf = StratifiedKFold(n_splits=f)
        for train, test in skf.split(data_s, label_s):
            TRid_ffold[count_ffold].append(train)
            Tsid_ffold[count_ffold].append(test)
            count_ffold += 1
        trainsubs_ffold += [s] * len(train)

    return TRid_ffold, Tsid_ffold, trainsubs_ffold, choose_sub

def indi_normalization(x, subjects, nb_persub):
    # x is the original data
    x_nor = []
    nb_subs = int(x.shape[0]/nb_persub)
    index_subjects = split_subjects(subjects)
    for m in range(nb_subs):
        data_m = StandardScaler().fit_transform(x[index_subjects[m]])
        x_nor.append(data_m)
    x_nor = np.vstack(x_nor)
    return x_nor

def labels_k(x, y, subjects, nb_persub, TRid_ffold, Tsid_ffold, r):
    # y is labels of the chosen k subjects
    index_subjects = split_subjects(subjects)
    k = int(x.shape[0] / nb_persub)
    f = int(1 / (1 - r))
    labelR_ffold = [[] for j in range(f)]
    labels_ffold = [[] for j in range(f)]
    for s in range(k):
        label_s = y[index_subjects[s]]
        for nb_ffold in range(f):
            labelR_ffold[nb_ffold] += list(label_s[TRid_ffold[nb_ffold][s]])
            labels_ffold[nb_ffold] += list(label_s[Tsid_ffold[nb_ffold][s]])
    return labelR_ffold, labels_ffold

def indi_nor_k(x, Dm, Ym, subjects, nb_persub, TRid_ffold, Tsid_ffold, r):
    # x is the original data of k chosen subjects
    # Dm is the original data of Dm
    x_nor = indi_normalization(x,subjects, nb_persub)
    index_subjects = split_subjects(subjects)
    k = int(x.shape[0]/nb_persub)
    f = int(1 / (1 - r))
    TR_ffold = [[] for j in range(f)]
    Ts_ffold = [[] for j in range(f)]
    for s in range(k):
        data_s = x_nor[index_subjects[s]]
        for nb_ffold in range(f):
            TR_ffold[nb_ffold].append(data_s[TRid_ffold[nb_ffold][s]])
            Ts_ffold[nb_ffold].append(data_s[Tsid_ffold[nb_ffold][s]])
    for j in range(f):
        TR_ffold[j] = np.vstack(TR_ffold[j])
        Ts_ffold[j] = np.vstack(Ts_ffold[j])
    Dm_nor = indi_normalization(Dm, subjects, nb_persub)
    Ym_nor = Ym
    return TR_ffold, Ts_ffold, Dm_nor, Ym_nor

def global_nor_k(x, Dm, Ym, subjects, nb_persub, TRid_ffold, Tsid_ffold, r):
    # x is the original data of k chosen subjects
    # Dm is the original data of Dm

    index_subjects = split_subjects(subjects)
    k = int(x.shape[0]/nb_persub)
    f = int(1 / (1 - r))
    TR_ffold = [[] for j in range(f)]
    Ts_ffold = [[] for j in range(f)]
    for s in range(k):
        data_s = x[index_subjects[s]]
        for nb_ffold in range(f):
            TR_ffold[nb_ffold].append(data_s[TRid_ffold[nb_ffold][s]])
            Ts_ffold[nb_ffold].append(data_s[Tsid_ffold[nb_ffold][s]])
    Tm = [[] for j in range(f)]
    for j in range(f):
        TR_ffold[j] = np.vstack(TR_ffold[j])
        Ts_ffold[j] = np.vstack(Ts_ffold[j])
        scaler = StandardScaler()
        scaler.fit(TR_ffold[j])
        TR_ffold[j] = scaler.transform(TR_ffold[j])
        Ts_ffold[j] = scaler.transform(Ts_ffold[j])
        Tm[j] = Dm[:]
        Tm[j] = scaler.transform(Tm[j])
    Dm_nor = Tm
    Ym_nor = Ym
    return TR_ffold, Ts_ffold, Dm_nor, Ym_nor

def part_nor_k(x, Dm, Ym, subjects, nb_persub, TRid_ffold, Tsid_ffold, r):
    # x is the original data of k chosen subjects
    # Dm is the original data of Dm

    index_subjects = split_subjects(subjects)
    k = int(x.shape[0]/nb_persub)
    f = int(1 / (1 - r))
    TR_ffold = [[] for j in range(f)]
    Ts_ffold = [[] for j in range(f)]
    for s in range(k):
        data_s = x[index_subjects[s]]
        for nb_ffold in range(f):
            TR_ffold[nb_ffold].append(StandardScaler(data_s[TRid_ffold[nb_ffold][s]]))
            Ts_ffold[nb_ffold].append(StandardScaler(data_s[Tsid_ffold[nb_ffold][s]]))
    for j in range(f):
        TR_ffold[j] = np.vstack(TR_ffold[j])
        Ts_ffold[j] = np.vstack(Ts_ffold[j])
    Dm_nor = []
    Ym_nor = []
    for m in range(int(Dm.shape[0] / nb_persub)):
        data_m = Dm[index_subjects[m]]
        label_m = Ym[index_subjects[m]]
        skf = StratifiedKFold(n_splits=f)
        for m_train, m_test in skf.split(data_m, label_m):
            Dm_nor.append(StandardScaler().fit_transform(data_m[m_test]))
            Ym_nor += list(label_m[m_test])
    Dm_nor = np.vstack(Dm_nor)
    return TR_ffold, Ts_ffold, Dm_nor, Ym_nor

def first_N_same_size(TR_ffold, label_ffold, trainsubs_ffold, r, k, trainsize):
    # each chosen subject only use N samples from the trainset to train clf
    # TR_ffold is the normalized data, label_ffold is TR_ffold's label
    index_subjects = split_subjects(trainsubs_ffold)
    N = int(trainsize / k)
    f = int(1 / (1 - r))
    TR_N_ffold = [[] for j in range(f)]
    labelR_N_ffold = [[] for j in range(f)]
    trainsubs_N_ffold = []
    for nb_ffold in range(f):
        data_f = TR_ffold[nb_ffold]
        label_f = label_ffold[nb_ffold]
        trainsubs_N_ffold = []
        for s in range(k):
            train_data = data_f[index_subjects[s]]
            train_label = label_f[s]
            ids_zero = [i for i, label in enumerate(train_label)
                       if label == 0][:math.ceil(N / 2)]
            ids_one = [i for i, label in enumerate(train_label)
                       if label == 1][:math.floor(N / 2)]
            ids_N = np.concatenate((ids_zero, ids_one))
            TR_N_ffold[nb_ffold].append(train_data[ids_N])
            labelR_N_ffold[nb_ffold] += list(train_label[ids_N])
            TR_N_ffold[nb_ffold] = np.vstack(TR_N_ffold[nb_ffold])
            trainsubs_N_ffold += np.unique(trainsubs_ffold)[s] * N

    return TR_N_ffold, labelR_N_ffold, trainsubs_N_ffold

def train_test_clf(TR, labelR_ffold, Ts, labels_ffold, Tm, Ym,
                   trainsubs_ffold, note, r, clf_method='logis'):
    # TR Ts Tm are normalizaed data, note includes all parameters for the title of saved data.
    Tm = np.asarray(Tm)
    logis = LogisticRegression
    svm = SVC
    logis_parameters = [{'penalty': ['l1', 'l2'],
                         'C': [10, 1, 0.1, 0.01, 0.005, 0.001, 0.0008,
                               0.0005, 0.0001, 0.00005, 0.00001]}]
    svm_parameters = [{'kernel': ['linear'],
                       'C': [10, 1, 0.1, 0.01, 0.005, 0.001, 0.0008,
                             0.0005, 0.0001, 0.00005, 0.00001]}]
    tuned_parameters = logis_parameters if clf_method == 'logis' else svm_parameters
    clf_ingrid = logis if clf_method == 'logis' else svm
    tol = 0.001 if clf_method == 'logis' else 0.01

    Ts_acc_ffold = []
    Tm_acc_ffold = []
    f = int(1 / (1 - r))
    for j in range(f):
        x_grid = TR[j]
        y_grid = labelR_ffold[j]
        subs_grid = trainsubs_ffold

        logo = LeaveOneGroupOut()
        grid_cv = list(logo.split(x_grid, y_grid, subs_grid))
        # gkf = GroupKFold(n_splits=10)
        # grid_cv = list(gkf.split(x_grid, y_grid, subs_grid))
        grid_clf = GridSearchCV(clf_ingrid(tol=tol), tuned_parameters,
                                cv=grid_cv, n_jobs=3)
        grid_clf.fit(x_grid, y_grid)
        print('best params', grid_clf.best_params_)
        joblib.dump(grid_clf.best_params_,
                    result_dir + 'gridtables/bestparam_{}_{}fold.pkl'.
                    format(note, j))
        # grid_csv = pd.DataFrame.from_dict(grid_clf.cv_results_)
        # with open(result_dir + 'gridtables/clfgrid_{}_{}fold.csv'.
        #         format(note, j), 'w') as f:
        #     grid_csv.to_csv(f)
        if clf_method == 'logis':
            choose_clf = clf_ingrid(C=grid_clf.best_params_['C'], tol=tol,
                                    penalty=grid_clf.best_params_['penalty'])
        else:
            choose_clf = clf_ingrid(C=grid_clf.best_params_['C'], tol=tol)

        choose_clf.fit(x_grid, y_grid)
        joblib.dump(choose_clf, result_dir +
                    'datasave/chosen_clf_{}_{}fold.pkl'.
                    format(note, j))

        pre_Ts = choose_clf.predict(Ts[j])
        score_Ts = accuracy_score(labels_ffold[j], pre_Ts)
        Ts_acc_ffold.append(score_Ts)
        if len(Tm.shape) > 2:
            pre_Tm = choose_clf.predict(Tm[j])
        else:
            pre_Tm = choose_clf.predict(Tm)
        score_Tm = accuracy_score(Ym, pre_Tm)
        Tm_acc_ffold.append(score_Tm)


    print('Ts acc', Ts_acc_ffold)
    print('Tm acc', Tm_acc_ffold)
    Ts_mean_ffold = np.mean(Ts_acc_ffold)
    Tm_mean_ffold = np.mean(Tm_acc_ffold)
    return Ts_mean_ffold, Tm_mean_ffold




def main():
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    args = sys.argv[1:]
    # data_method = args[0]  # indi, global, no
    # clf_method = args[1]  # logis, svm
    # experiment = args[2]
    # r = float(args[3])
    # k = int(args[4])
    # trainsize = int(args[5])
    # nrounds = 10


    data_method = 'no'  # indi, global, no
    clf_method = 'logis'  # logis, svm
    experiment = 'meg'
    r = 0.875
    k = 5
    trainsize = 1290
    almost_size = 1300
    nrounds = 2

    if experiment == 'fmril':
        source = fmril
    elif experiment == 'fmrir':
        source = fmrir
    else:
        source = meg

    note = 'experiment2_same_size_{}_{}_{}_{}r_{}subs_{}trainsize'.format(experiment,
                                                                   clf_method, data_method,
                                                                   r, k,trainsize)

    print(note)
    y_target = source.y_target
    x_data = source.x_data_pow if experiment == 'meg' else source.x_data
    x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
    x = x_indi if data_method == 'indi' else x_data
    subjects = source.subjects
    nb_persub = source.nb_samples_each

    if trainsize % k == 0 and trainsize/k <= nb_persub*r:
        results = same_training_size(x, y_target, subjects, nb_persub, r, k,
                                     trainsize, clf_method, experiment, data_method,
                                          nrounds)
        np.savez(result_dir + 'acc_{}_{}_{}_{}r_{}subs_{}trainsize_{}rounds_10splits.npz'.
                 format(experiment, clf_method, data_method, r, k, trainsize, nrounds),
                 Ts1_acc=results[0], Tm1_acc=results[1], Ts2_acc=results[2], Tm2_acc=results[3])
        np.savez(result_dir + 'acc_{}_{}_{}_{}r_{}subs_{}almost_trainsize_{}rounds_10splits.npz'.
                 format(experiment, clf_method, data_method, r, k, almost_size, nrounds),
                 Ts1_acc=results[0], Tm1_acc=results[1], Ts2_acc=results[2], Tm2_acc=results[3])

    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


if __name__ == '__main__':
    main()
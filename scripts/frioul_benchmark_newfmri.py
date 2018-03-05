import os
import sys
main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

from data import fmri_localizer as newfmri
from data import fmri_data_cv as fmril
from model import utils

from sklearn.externals import joblib
import numpy as np
import time

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
def main():

    args = sys.argv[1:]
    experiment = args[0]
    nor_method = args[1]
    clf_method = args[2]
    nb_split = int(args[3])
    # experiment = 'newfmri'
    # nor_method = 'indi'
    # clf_method = 'logis'
    # nb_split = 1

    result_dir = main_dir + '/results'
    result_dir = result_dir + '/{}/bench'.format(experiment)

    if experiment == 'newfmri':
        source = newfmri
        cv = np.load(main_dir + '/data/cv_newfmri.npy')[:200]
    elif experiment == 'fmril':
        source = fmril
        cv = source.cross_v

    y_target = np.array(source.y_target)
    subjects = np.array(source.subjects)
    x_data = np.asarray(source.x_data)
    x_indi = np.asarray(source.x_indi)
    x = x_indi if nor_method == 'indi' else x_data

    train, test = cv[nb_split]
    x_train = x[train]
    y_train = y_target[train]
    x_test = x[test]
    y_test = y_target[test]
    subjects_train = subjects[train]
    name = '{}_{}_{}_split{}_banchmark'.format(experiment, nor_method,
                                                clf_method, nb_split)
    print(name)
    result = {}
    grid_score, best_params, pre = utils.n2n_gridsearch(x_train, y_train, x_test, y_test,
                                              subjects_train, clf_method, K=10, grid=False)

    result['score'] = grid_score
    result['params'] = best_params
    result['pre'] = pre
    if best_params['C'] in [10, 0.00001]:
        print('C is on the boundary', best_params['C'])
        joblib.dump(result, result_dir + '/C/{}.pkl'.format(name))
    print('best params for gridsearch', best_params)
    print('score for split {}'.format(nb_split), grid_score)
    joblib.dump(result, result_dir + '/{}.pkl'.format(name))


if __name__ == '__main__':
    main()
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

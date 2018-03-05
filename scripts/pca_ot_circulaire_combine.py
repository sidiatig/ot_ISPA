import numpy as np
import itertools
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import os
import scipy.stats as stats

from data import fmri_data_cv as fmril
from data import fmri_localizer as newdata

experiment = 'fmril'
nor_method = 'indi'
clf_method = 'logis'
ot_method = 'l1l2'
dim = 40

source = fmril if experiment == 'fmril' else newdata
y = source.y_target
id_subs = source.index_subjects

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
result_dir = result_dir + '/pca_ot_circulaire/{}'.format(experiment)

clf_means = []
base_means = []
target_list = [i for i in range(10)]
for id in range(len(target_list)):
    target_subs = target_list[id]
    pre_matrix = []
    base_accs = []
    source_subs = np.delete(target_list, id)
    for sources in list(itertools.combinations(source_subs, len(source_subs)-1)):
        sources = list(sources)
        note = '{}_{}_{}_{}_{}source_{}target_{}pca_ot_circulaire'.format(experiment,
                                                                           nor_method,
                                                                           clf_method,
                                                                           ot_method,
                                                                           sources,
                                                                           target_subs, dim)

        file = joblib.load(result_dir + '/{}.pkl'.format(note))
        pre_matrix.append(file['target_pre'])
        base_accs.append(file['clf_base_acc'][1])
    pre_matrix = np.vstack(pre_matrix)
    pre_mean = np.mean(pre_matrix, 0)
    print('pre_mean for {} target'.format(id))
    print(pre_mean)
    pre = pre_mean >= 0.5
    print(pre)
    y_true = y[id_subs[id]]
    mean_score = accuracy_score(y_true, pre)
    print('mean clf score', mean_score)
    print('mean base score', np.mean(base_accs))
    clf_means.append(mean_score)
    base_means.append(np.mean(base_accs))
print('clf_means', np.mean(clf_means),clf_means)
print('base_means', np.mean(base_means),base_means)
stats_resutls = stats.ttest_rel(base_means, clf_means)
print(stats_resutls)


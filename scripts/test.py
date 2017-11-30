import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as models
import ot

from sklearn.externals import joblib
import numpy as np
import time

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

args = sys.argv[1:]
experiment = 'fmril'
nor_method = 'indi'
clf_method = 'logis'
source_id = 1
target_id = 2
op_function = 'l1l2'
note = '{}s_{}t_{}_{}_{}_skinkhorn_{}'.format(source_id, target_id,
                                                   experiment, nor_method, clf_method,
                                                 op_function)
print(note)

if experiment == 'fmril':
    source = fmril
elif experiment == 'fmrir':
    source = fmrir
else:
    source = meg

y_target = source.y_target
subjects = np.array(source.subjects)
nb_samples_each = source.nb_samples_each
x_data = source.x_data_pow if experiment == 'meg' else source.x_data
x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
x = x_indi if nor_method == 'indi' else x_data

i = 1
j = 2
indices_sub = models.split_subjects(subjects)
xs = x[indices_sub[i]]
ys = y_target[indices_sub[i]]
xt = x[indices_sub[j]]
yt = y_target[indices_sub[j]]
M = ot.utils.dist(xs,xt)
M = ot.utils.cost_normalization(M, 'max')

a = np.ones(xs.shape[0])/xs.shape[0]
b = np.ones(xt.shape[0])/xt.shape[0]


trans_s = []
Gs = []
distances = []
# for reg in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
#     for eta in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:
for reg in [1e-3]:
    for eta in [1e-4]:
        if op_function == 'lpl1':
            G = ot.da.sinkhorn_lpl1_mm(a=a, labels_a=ys, b=b, M=M, reg=reg, eta=eta)
            xst = len(a) * np.dot(G, xt)
        elif op_function == 'l1l2':
            G= ot.da.sinkhorn_l1l2_gl(a=a, labels_a=ys, b=b, M=M, reg=reg, eta=eta)
            xst = len(a) * np.dot(G, xt)
        print('xt',xt)
        print('xst', xst)

# np.savez(result_dir + '/1s_2t_lpl1.npz',distances=distances, transformations = trans_s, Gs=Gs)

# skf = StratifiedKFold(n_splits=10)
# grid_cv = list(skf.split(xs, ys))
# print(grid_cv)

# clf = LogisticRegression(C=0.001, penalty='l2')
# clf.fit(xs,ys)
# pre = clf.predict(xt)
# score = accuracy_score(yt,pre)
# print('score of logis',score)
# ori_best, ori_grid, ori_score = models.gridsearch_persub(xs,ys,xt,yt,experiment=experiment,clf_method=clf_method)
# print('ori_score', ori_score)
# print('ori_best',ori_best)
# print('ori_grid', ori_grid)

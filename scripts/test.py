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
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import numpy as np
import time

import ot

import matplotlib.pyplot as plt

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

# args = sys.argv[1:]
# experiment = args[0]
# nor_method = args[1]
# clf_method = args[2]
# source_id = int(args[3])
# target_id = int(args[4])
# op_function = args[5] # 'lpl1' 'l1l2'

metric_xst = 'null'

experiment = 'fmril'
nor_method = 'no'
clf_method = 'logis'
source_id = 1
target_id = 2
op_function = 'l1l2' # 'lpl1' 'l1l2'

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
x = x_indi.copy() if nor_method == 'indi' else x_data.copy()

indices_sub = fucs.split_subjects(subjects)
nb_subs = len(indices_sub)

i = source_id
j = target_id
xs = x[indices_sub[i]]
ys = y_target[indices_sub[i]]
xt = x[indices_sub[j]]
yt = y_target[indices_sub[j]]
x_pos = np.vstack((xs[ys==1],xt[yt==1]))
x_neg = np.vstack((xs[ys==0],xt[yt==0]))
b_pos = np.empty(x_pos.shape)
b_neg = np.empty(x_neg.shape)

for i in range(x_pos.shape[0]):
    data = x_pos[i] - np.min(x_pos[i])
    data /= np.sum(data)
    b_pos[i] = data
for i in range(x_neg.shape[0]):
    data = x_neg[i] - np.min(x_neg[i])
    data /= np.sum(data)
    b_neg[i] = data
b_pos = np.transpose(b_pos)
b_neg = np.transpose(b_neg)
# pos_distributions = b_pos.shape[1]
# neg_distributions = b_neg.shape[1]
# loss matrix + normalization
M_pos = ot.utils.dist0(b_pos.shape[0])
M_pos = np.sqrt(M_pos)
M_pos /= np.median(M_pos)
M_neg = ot.utils.dist0(b_neg.shape[0])
M_neg = np.sqrt(M_neg)
M_neg /= np.median(M_neg)
alpha = 1/b_neg.shape[1]  # 0<=alpha<=1
weights = np.ones(b_neg.shape[1]) * alpha
# l2bary
# bary_l2 = B.dot(weights)
# wasserstein
reg = 1/800
bw_pos, log = ot.bregman.barycenter(b_pos, M_pos, reg, weights,
                                       numItermax=100, log=True)
bw_neg, log = ot.bregman.barycenter(b_neg, M_neg, reg, weights,
                                       numItermax=100, log=True)
for j in range(3,10):
    x_test = xs = x[indices_sub[j]]
    y_test = y_target[indices_sub[j]]
    dists=[]
    for i in range(x_test.shape[0]):
        dist_pos = np.linalg.norm(x_test[i] - bw_pos)
        dist_neg = np.linalg.norm(x_test[i] - bw_neg)
        print(dist_pos, dist_neg)
        dists.append(1) if dist_pos<dist_neg else dists.append(0)

    acc = accuracy_score(y_test, dists)
    print(j, acc)

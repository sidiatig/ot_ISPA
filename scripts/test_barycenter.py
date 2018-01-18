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
import numpy as np
import time
import scipy.io
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

A = fmril.x_data[:80]
B = np.empty(A.shape)
for i in range(A.shape[0]):
    data = A[i] - np.min(A[i])
    data /= np.sum(data)
    B[i] = data
B = np.transpose(B)
n_distributions = B.shape[1]
# loss matrix + normalization
M = ot.utils.dist0(B.shape[0])
M = np.sqrt(M)
M /= np.median(M)
alpha = 1/B.shape[1]  # 0<=alpha<=1
weights = np.ones(B.shape[1]) * alpha
# l2bary
bary_l2 = B.dot(weights)
# wasserstein
reg = 1/800
bary_wass, log = ot.bregman.barycenter(B, M, reg, weights,
                                       numItermax=100, log=True)
print(log)
np.savez(result_dir + '/fmril_no_1s_2t_iter100_reg800.npz', bary_l2=bary_l2, bary_wass=bary_wass)


arti_data = scipy.io.loadmat('/hpc/crise/wang.q/data/artificial_data.mat')
arti = arti_data['x']
arti = arti.reshape((80, 60*90))

arti_dis = np.empty(arti.shape)
for i in range(arti.shape[0]):
    data = arti[i] - np.min(arti[i])
    data /= np.sum(data)
    arti_dis[i] = data
arti_dis = np.transpose(arti_dis)
M = ot.utils.dist0(arti_dis.shape[0])
M = np.sqrt(M)
M /= np.median(M)
alpha = 1/arti_dis.shape[1]  # 0<=alpha<=1
weights = np.ones(arti_dis.shape[1]) * alpha
bary_l2 = arti_dis.dot(weights)
# wasserstein
reg = 1/800
bary_wass, log = ot.bregman.barycenter(arti_dis, M, reg, weights,
                                       numItermax=100, log=True)

import os
import sys
import matplotlib
matplotlib.use('Agg')

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as fucs
from model import dual_optima_barycenter as dual

from sklearn.externals import joblib
import scipy.stats as stats
import numpy as np
import time
import ot

import matplotlib.pyplot as plt

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'


experiment = 'fmril'
nor_method = 'indi'
dist_type = 'vector'
reg = 0.001
ids = [i for i in range(2)]
ite = 1000
pos_neg = 'neg'

if experiment == 'fmril':
    source = fmril
elif experiment == 'fmrir':
    source = fmrir
else:
    source = meg

result_dir = result_dir + '/{}'.format(experiment)
y_target = source.y_target
subjects = np.array(source.subjects)
ids_subs = source.index_subjects
x_data = source.x_data_pow if experiment == 'meg' else source.x_data
x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
x = x_indi.copy() if nor_method == 'indi' else x_data.copy()

for source_id in ids:
    note = '{}_{}_sub{}_{}_{}reg_{}ite_{}'.format(experiment, nor_method,
                                                  source_id, dist_type, reg, ite, pos_neg)
    print(note)
    A = x[ids_subs[source_id]][:20] if pos_neg == 'pos' else x[ids_subs[source_id]][20:40]
    B = np.empty(A.shape)
    for i in range(A.shape[0]):
        data = A[i] - np.min(A[i])
        data /= np.sum(data)
        B[i] = data
    B = np.transpose(B)
    n_distributions = B.shape[1]
    # loss matrix + normalization
    if dist_type == 'vector':
        M = ot.utils.dist0(B.shape[0])
        M = np.sqrt(M)
        M /= np.median(M)
    else:
        M_image = dual.distance_matrix(60, 90)
        M = M_image/(np.max(M_image) * 0.5)
    alpha = 1/B.shape[1]  # 0<=alpha<=1
    weights = np.ones(B.shape[1]) * alpha
    # l2bary
    bary_l2 = B.dot(weights)
    # wasserstein

    bary_wass, log = ot.bregman.barycenter(B, M, reg, weights,
                                           numItermax=ite, stopThr=1e-4, log=True)
    print(log)
    np.savez(result_dir + '/{}.npz'.format(note), bary_l2=bary_l2, bary_wass=bary_wass)
    print('barycenter min, max', bary_wass.min(), bary_wass.max())
    plt.figure()
    fig = plt.gcf()
    fig.suptitle('{}'.format(note), fontsize=14)
    plt.subplot(1,2,1)
    plt.imshow(bary_wass.reshape((60,90)))
    plt.title('wass bary')
    plt.subplot(1,2,2)
    plt.imshow(bary_l2.reshape((60,90)))
    plt.title('l2 bary')
    plt.savefig(result_dir+'/{}.png'.format(note))
    plt.close()
#
# arti_data = scipy.io.loadmat('/hpc/crise/wang.q/data/artificial_data.mat')
# arti = arti_data['x']
# arti = arti.reshape((80, 60*90))
#
# arti_dis = np.empty(arti.shape)
# for i in range(arti.shape[0]):
#     data = arti[i] - np.min(arti[i])
#     data /= np.sum(data)
#     arti_dis[i] = data
# arti_dis = np.transpose(arti_dis)
# M = ot.utils.dist0(arti_dis.shape[0])
# M = np.sqrt(M)
# M /= np.median(M)
# alpha = 1/arti_dis.shape[1]  # 0<=alpha<=1
# weights = np.ones(arti_dis.shape[1]) * alpha
# bary_l2 = arti_dis.dot(weights)
# # wasserstein
# reg = 1/800
# bary_wass, log = ot.bregman.barycenter(arti_dis, M, reg, weights,
#                                        numItermax=100, log=True)

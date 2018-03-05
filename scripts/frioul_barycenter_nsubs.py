import os
import sys
import matplotlib
matplotlib.use('Agg')

main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as fucs
from model import dual_optima_barycenter as dual

import numpy as np
import time
import ot
import matplotlib.pyplot as plt



def main():
    args = sys.argv[1:]
    experiment = args[0]
    nor_method = args[1]
    dist_type = args[2]
    id1 = int(args[3])
    id2 = int(args[4])
    reg = 1/1000
    ite = 500

    ids = (id1, id2)
    # experiment = 'fmril'
    # nor_method = 'indi'
    # dist_type = 'matrix'
    # reg = 1/1000
    # ids = (1,2)
    # ite = 500


    if experiment == 'fmril':
        source = fmril
    elif experiment == 'fmrir':
        source = fmrir
    else:
        source = meg

    result_dir = main_dir + '/results/barycenter_clf/{}'.format(experiment)
    fig_dir = main_dir + '/figs/barycenter_clf/{}'.format(experiment)

    y_target = source.y_target
    ids_subs = source.index_subjects
    x_data = source.x_data_pow if experiment == 'meg' else source.x_data
    x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
    x = x_indi.copy() if nor_method == 'indi' else x_data.copy()

    note = '{}_{}_sub{}_{}_{}reg_{}ite'.format(experiment, nor_method, ids, dist_type, reg, ite)
    print(note)
    data_pos = []
    data_neg = []
    for id in ids:
        x_target = x[ids_subs[id]]
        y_label = y_target[ids_subs[id]]
        data_pos.append(x_target[y_label==1])
        data_neg.append(x_target[y_label==0])
    data_pos = np.vstack(data_pos)
    data_neg = np.vstack(data_neg)

    set_id = 0
    for set in [data_pos, data_neg]:
        pos_neg = 'pos' if set_id == 0 else 'neg'
        A = set
        B = np.empty(A.shape)
        for i in range(A.shape[0]):
            data = A[i] - np.min(A[i])
            data /= np.sum(data)
            B[i] = data
        B = np.transpose(B)

        # loss matrix + normalization
        if dist_type == 'vector' or experiment == 'newdata':
            M = ot.utils.dist0(B.shape[0])
            M = np.sqrt(M)
            M /= np.median(M)
        else:
            M_image = dual.distance_matrix(60, 90)
            M = M_image/(np.max(M_image) * 0.5)
        alpha = 1/B.shape[1]  # 0<=alpha<=1
        weights = np.ones(B.shape[1]) * alpha

        bary_l2 = B.dot(weights)
        bary_wass, log = ot.bregman.barycenter(B, M, reg, weights,
                                               numItermax=ite, log=True)
        print('err',log)
        print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

        np.savez(result_dir + '/{}_{}.npz'.format(note, pos_neg), bary_l2=bary_l2, bary_wass=bary_wass)
        print('barycenter min, max', bary_wass.min(), bary_wass.max())
        plt.figure()
        fig = plt.gcf()
        fig.suptitle('{}_{}'.format(note, pos_neg), fontsize=14)
        plt.subplot(1,2,1)
        plt.imshow(bary_wass.reshape((60,90)))
        plt.title('wass bary')
        plt.subplot(1,2,2)
        plt.imshow(bary_l2.reshape((60,90)))
        plt.title('l2 bary')
        plt.savefig(fig_dir+'/{}_{}.png'.format(note, pos_neg))
        plt.close()
        set_id += 1

if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    main()
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

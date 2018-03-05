import matplotlib

matplotlib.use('Agg')

import os
import sys

main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

import numpy as np
import matplotlib.pyplot as plt
from model import dual_optima_barycenter as dual
from data import fmri_localizer as newdata
from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
import ot
import time

# main_dir = main_dir + '/ot_ISPA'
result_dir = main_dir + '/results/barycenter_clf'
fig_dir = main_dir + '/figs/barycenter_clf'


# 2 subjects, each subject has one zone,
# uniform noise:  (b - a) * random_sample(size) + a
# noise is uniformly distributed in the interval [a,b)
# gaussian noise: np.random.normal(mu, sigma, size)

def main():
    # args = sys.argv[1:]
    # reg = float(args[0])
    # id = int(args[1])  # nb_samples is the nb of samples in each subject
    # experiment = args[2]
    # data_method = args[3]


    reg = 0.01
    id = 0  # nb_samples is the nb of samples in each subject
    experiment = 'fmril'
    data_method = 'indi'

    note = "{}_{}_barycenter_subject{}".format(experiment, data_method, id)
    print(note)

    if experiment == 'newdata':
        source = newdata
    elif experiment == 'fmril':
        source = fmril
    else:
        source = fmrir
    x_indi = source.x_indi
    x_data = source.x_data
    x = x_indi if data_method == 'indi' else x_data
    y_target = source.y_target
    subjects = source.subjects
    ids_subs = source.index_subjects

    x_target = x[ids_subs[id]]
    y_label = y_target[ids_subs[id]]
    x_pos = x_target[y_label == 1]
    x_neg = x_target[y_label == 0]
    count = 0
    for dataset in [x_pos, x_neg]:
        arti_dis = np.empty(dataset.shape)
        for i in range(dataset.shape[0]):
            data = dataset[i] - np.min(dataset[i])
            data /= np.sum(data)
            arti_dis[i] = data
        A = np.transpose(arti_dis)  # A is the matrix of histograms

        M_image = ot.utils.dist0(A.shape[0])
        alpha = 1 / A.shape[1]  # 0<=alpha<=1
        weights = np.ones(A.shape[1]) * alpha
        barycenter, log = ot.bregman.barycenter(A, M_image, reg, weights,
                                           numItermax=1000, stopThr=1e-4, log=True)
        print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
        if count == 0:
            bary_pos = barycenter
        else:
            bary_neg = barycenter
    np.savez(result_dir + '/{}.npz'.format(note), bary_pos=bary_pos, bary_neg=bary_neg)
    if experiment != 'newdata':
        vmax = 0.0005
        plt.figure(figsize=(9, 5))
        fig = plt.gcf()
        fig.suptitle('{}'.format(note), fontsize=14)
        plt.subplot(1, 5, 1)
        plt.imshow(bary_pos.reshape((60,90)), vmin=0, vmax=vmax)
        plt.title('bary_pos')
        plt.subplot(1, 5, 2)
        plt.imshow(bary_neg.reshape((60,90)), vmin=0, vmax=vmax)
        plt.title('bary_neg')
        plt.savefig(fig_dir + '/{}_{}vmax.png'.format(note, vmax))



if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    main()
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


# bregman or dual barycenter, matrix distance
# values of pixels in active zones are set to be gaus with mean of 5, std=1,
# four experiments:
# 1 two subs ,each one has N images with the same active location
# 2 one sub has N images with one slightly different active location
# 3 two subs, each one has N images with one slightly different active location
# 4 two subs, each one has N images with two slightly different active locations
import matplotlib
matplotlib.use('Agg')

import os
import sys
main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from model import dual_optima_barycenter as dual
import time


# main_dir = main_dir + '/ot_ISPA'
result_dir = main_dir  + '/results/dual_barycenter/artificial_active1'
fig_dir = main_dir + '/figs/dual_barycenter/artificial_active1'
# 2 subjects, each subject has one zone,
# uniform noise:  (b - a) * random_sample(size) + a
# noise is uniformly distributed in the interval [a,b)
# gaussian noise: np.random.normal(mu, sigma, size)

def main():
    args = sys.argv[1:]
    noise_type = args[0]
    reg = float(args[1])
    c = float(args[2])  # step for gradient
    noise = float(args[3])
    nb_samples = int(args[4]) # nb is the nb of samples in each subject or the number of subjects
    nb_subs = int(args[5])
    nb_zones = int(args[6])
    Ns = float(args[7]) # N is the weight of distance matrix
    loc_type = args[8]  # 'fixed' or 'flex
    bary_type = args[9]

    # noise_type = 'gaus'
    # reg = 0.001
    # c = -4  # step for gradient
    # noise = 0.001
    # nb_samples = 2  # nb_samples is the nb of samples in each subject
    # bary_type = 'bregman'
    # nb_subs = 2
    # nb_zones = 1
    # loc_type = 'fixed'  # 'fixed' or 'flex
    # Ns = 0.5  # N is the weight of distance matrix

    note = "artificial_gaussignal_{}_{}noise_{}subs_{}samples_{}zones" \
           "_{}reg_{}c_{}Ns_{}loc_matrixdist_{}".format(noise_type, noise, nb_subs,
                                       nb_samples, nb_zones,
                                       reg, c, Ns, loc_type, bary_type)
    print(note)

    x_min = [10, 70]
    x_max = [20, 80]
    y_min = [5, 41]
    y_max = [10, 46]

    x_size = 100
    y_size = 50

    mean = 0
    mu = 5
    std = 1


    noisefrees = np.zeros([2 * nb_samples, x_size, y_size])
    patterns = np.zeros([2 * nb_samples, x_size, y_size])
    if loc_type == 'fixed':  # fixed, 2 subs, one zone
        for sub_id in range(nb_subs):
            for sample_id in range(nb_samples):
                noisefrees[sample_id + sub_id * nb_samples][x_min[sub_id]:x_max[sub_id],
                y_min[sub_id]:y_max[sub_id]] = \
                    np.random.normal(mu, std, (x_max[sub_id]-x_min[sub_id],
                                               y_max[sub_id]-y_min[sub_id]))
                patterns[sample_id + sub_id * nb_samples] = \
                    noisefrees[sample_id + sub_id * nb_samples] + \
                    np.random.normal(mean, noise, (x_size, y_size))
    else:
        for sub_id in range(2):
            for sample_id in range(nb_samples):
                bias1 = int(2 * np.random.randn())
                bias2 = int(2 * np.random.randn())
                noisefrees[sample_id + sub_id * nb_samples][x_min[sub_id] + bias1:x_max[sub_id] + bias1,
                y_min[sub_id] + bias2:y_max[sub_id] + bias2] = \
                    np.random.normal(mu, std, (x_max[sub_id]-x_min[sub_id],
                                               y_max[sub_id]-y_min[sub_id]))

        if nb_subs == 2:  # flexible, 2subs, one zone
            for sub_id in range(nb_subs):
                for sample_id in range(nb_samples):
                    patterns[sample_id + sub_id * nb_samples] = \
                        noisefrees[sample_id + sub_id * nb_samples] + \
                        np.random.normal(mean, noise, (x_size, y_size))
        else:
            patterns = np.zeros([1 * nb_samples, x_size, y_size])
            if nb_zones == 1:  # flexible, one sub, one zone
                for sample_id in range(nb_samples):
                    patterns[sample_id] = noisefrees[sample_id] + \
                                          np.random.normal(mean, noise, (x_size, y_size))
            else:  # flexible, one sub, two zones
                noisefrees2 = np.zeros([1 * nb_samples, x_size, y_size])
                for sample_id in range(nb_samples):
                    noisefrees2[sample_id] = noisefrees[sample_id + nb_samples] + noisefrees[sample_id]
                    patterns[sample_id] = noisefrees2[sample_id] + \
                                          np.random.normal(mean, noise, (x_size, y_size))
    arti = patterns.reshape([-1, x_size*y_size])
    arti_dis = np.empty(arti.shape)
    for i in range(arti.shape[0]):
        data = arti[i] - np.min(arti[i])
        data /= np.sum(data)
        arti_dis[i] = data
    A = np.transpose(arti_dis) # A is the matrix of histograms

    M_image = dual.distance_matrix(x_size, y_size)
    M_image1 = M_image / (np.max(M_image) * Ns)

    alpha = 1 / (A.shape[1])  # 0<=alpha<=1
    weights = np.ones(A.shape[1]) * alpha
    bary_eucli = A.dot(weights)
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    if bary_type == 'dual':
        barycenter = dual.entropic_barycenters(A, M_image1, reg=reg, c=c,
                                                    max_iter=1000, tol=1e-6, log=False)
    else:
        barycenter = dual.bregman_barycenter(A, M_image1, reg, weights,
                                                numItermax=1000, stopThr=1e-6, log=False)
    print('image min max', patterns.min(), patterns.max())
    print('histograms min max', A.min(), A.max())
    print('barycenter euclidean min max', bary_eucli.min(), bary_eucli.max())
    print('barycenter {} min max'.format(bary_type), barycenter.min(), barycenter.max())
    np.savez(result_dir + '/{}.npz'.format(note), patterns=patterns, A=A, bary_eucli=bary_eucli,
             barycenter=barycenter)

    # original data
    plt.figure(figsize=(10, 3))
    fig = plt.gcf()
    fig.suptitle('{}'.format(note), fontsize=14)
    for i in range(patterns.shape[0]):
        plt.subplot(1, patterns.shape[0] + 1, i + 1)
        plt.imshow(patterns[i], vmin=0, vmax=7)
    plt.savefig(fig_dir + '/{}_ori.png'.format(note))

    vmax = 0.005
    plt.figure(figsize=(10,5))
    fig = plt.gcf()
    fig.suptitle('{}'.format(note), fontsize=14)
    plt.subplot(1, 4, 1)
    plt.imshow(patterns[0], vmin=0, vmax=7)
    plt.title('image 1')
    plt.subplot(1, 4, 2)
    plt.imshow(patterns[-1], vmin=0, vmax=7)
    plt.title('image 2')
    plt.subplot(1, 4, 3)
    plt.imshow(bary_eucli.reshape(patterns[0].shape), vmin=0, vmax=vmax)
    plt.title('barycenter\neuclidean')
    plt.subplot(1, 4, 4)
    im = plt.imshow(barycenter.reshape(patterns[0].shape), vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('barycenter\n{}'.format(bary_type))
    plt.savefig(fig_dir + '/{}_{}vmax.png'.format(note, vmax))
    print(note)


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    main()
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


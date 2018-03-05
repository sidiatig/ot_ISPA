import matplotlib
matplotlib.use('Agg')

import os
import sys
main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

import numpy as np
import matplotlib.pyplot as plt
from model import dual_optima_barycenter as dual
import ot
import time


# main_dir = main_dir + '/ot_ISPA'
result_dir = main_dir  + '/results/dual_barycenter/artificial'
fig_dir = main_dir + '/figs/dual_barycenter/artificial'
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
    nb_samples = int(args[4]) # nb_samples is the nb of samples in each subject
    bary_type = args[5]
    Ns = float(args[6]) # N is the weight of distance matrix

    # noise_type = 'gaus'
    # reg = 0.001
    # c = -4  # step for gradient
    # noise = 0.001
    # nb_samples = 1  # nb_samples is the nb of samples in each subject
    # bary_type = 'bregman'
    # Ns = 0.5  # N is the weight of distance matrix


    note = "artificial_{}_{}noise_{}samples_{}reg_{}c_{}Ns_{}".format(noise_type,
                                                                     noise, nb_samples,
                                                                     reg, c, Ns, bary_type)
    print(note)

    x_min = [10, 70]
    x_max = [20, 80]
    y_min = [5, 41]
    y_max = [10, 46]

    x_size = 100
    y_size = 50

    mean = 0
    nb_subs = 2

    noisefrees = np.zeros([nb_subs, x_size, y_size])
    for sub_id in range(nb_subs):
        noisefrees[sub_id][x_min[sub_id]:x_max[sub_id], y_min[sub_id]:y_max[sub_id]] = 1

    patterns = np.zeros([nb_subs * nb_samples, x_size, y_size])
    for sub_id in range(nb_subs):
        for sample_id in range(nb_samples):
            if noise_type == 'uniform':
                patterns[sample_id + sub_id * nb_samples] = noisefrees[sub_id] + \
                                                            2 * noise * np.random.random_sample((x_size, y_size)) - noise
            else:
                patterns[sample_id + sub_id * nb_samples] = noisefrees[sub_id] + \
                                                            np.random.normal(mean, noise, (x_size, y_size))

    arti = patterns.reshape([-1, x_size*y_size])
    arti_dis = np.empty(arti.shape)
    for i in range(arti.shape[0]):
        data = arti[i] - np.min(arti[i])
        data /= np.sum(data)
        arti_dis[i] = data
    A = np.transpose(arti_dis) # A is the matrix of histograms

    # M = ot.utils.dist0(A.shape[0])
    # M = np.sqrt(M)
    # M_image1 = M / np.max(M)

    M_image = dual.distance_matrix(x_size, y_size)
    M_image1 = M_image / (np.max(M_image) * Ns)

    # a = np.ones(5000, dtype=np.float64)/ 5000
    # optima = dual.sinkhorn(a, A[:,1], M_image1, reg)[1]
    alpha = 1 / (nb_subs * nb_samples)  # 0<=alpha<=1
    weights = np.ones(nb_subs * nb_samples) * alpha
    bary_eucli = A.dot(weights)
    print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
    if bary_type == 'dual':
        barycenter, log = dual.entropic_barycenters(A, M_image1, reg=reg, c=c,
                                                    max_iter=1000, tol=1e-8, log=True)
    else:
        barycenter, log = ot.bregman.barycenter(A, M_image1, reg, weights,
                                                numItermax=1000, stopThr=1e-5, log=True)
    print('image min max', patterns[0].min(), patterns[0].max())
    print('histograms min max', A.min(), A.max())
    print('barycenter euclidean min max', bary_eucli.min(), bary_eucli.max())
    print('barycenter {} min max'.format(bary_type), barycenter.min(), barycenter.max())
    np.savez(result_dir + '/{}.npz'.format(note), patterns=patterns, A=A, bary_eucli=bary_eucli,
             barycenter=barycenter)


    vmax = 0.005
    plt.figure(figsize=(10,5))
    fig = plt.gcf()
    fig.suptitle('{}'.format(note), fontsize=14)
    plt.subplot(1, 4, 1)
    plt.imshow(patterns[0], vmin=0, vmax=1.5)
    plt.title('subject 1')
    plt.subplot(1, 4, 2)
    plt.imshow(patterns[nb_samples], vmin=0, vmax=1.5)
    plt.title('subject 2')
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


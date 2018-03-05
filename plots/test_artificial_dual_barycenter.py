import numpy as np
import ot
import matplotlib.pyplot as plt

import os
import sys
main_dir = os.path.split(os.getcwd())[0]
sys.path.append(main_dir)

from model import dual_optima_barycenter as dual
# main_dir = main_dir + '/ot_ISPA'
fig_dir = main_dir  + '/figs/distance_matrix'
# 2 subjects, each subject has one zone,
# uniform noise:  (b - a) * random_sample(size) + a
# noise is uniformly distributed in the interval [a,b)
# gaussian noise: np.random.normal(mu, sigma, size)

x_min = [10, 70]
x_max = [20, 80]
y_min = [5, 40]
y_max = [10, 45]

x_size = 100
y_size = 50

Ns = 0.5  # N is the weight of distance matrix
reg = 1 / 800
nb_subs = 2


# M with 2d coordinates
nx, ny = x_size, y_size
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(y,x)
coors = np.vstack((xv.flatten(), yv.flatten())).T
coor = np.empty(coors.shape)
coor[:,0] = coors[:,1]
coor[:,1] = coors[:,0]
M_image = ot.utils.dist(coor)
# M = np.sqrt(M)
M_image1 = M_image / (np.max(M_image) * Ns)

noisefrees = np.zeros([nb_subs, x_size, y_size])
for sub_id in range(nb_subs):
    noisefrees[sub_id][x_min[sub_id]:x_max[sub_id], y_min[sub_id]:y_max[sub_id]] = 1
plt.figure()
plt.subplot(1,2,1)
plt.imshow(noisefrees[0])
plt.subplot(1,2,2)
plt.imshow(noisefrees[1])

# vmax is the max for imshow of barycenter
vmax = 0.0005
noise_type = 'gaus'
# noises = np.arange(0.02, 0.03, 0.005)
noises = [0.02]
mean = 0
plt.figure()
fig = plt.gcf()  # Get a reference to the current figure.
fig.suptitle('two subjects with several samples, {} noise'.format(noise_type), fontsize=14)
j = 0
nbs = [i for i in range(1,2)]
for noise in noises:
    # nb_samples is nb of samples in each sub
    for nb_samples in nbs:
        alpha = 1 / (nb_subs * nb_samples)  # 0<=alpha<=1
        weights = np.ones(nb_subs * nb_samples) * alpha

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

        # a = np.ones(5000, dtype=np.float64)/ 5000
        # optima = dual.sinkhorn(a, A[:,1], M_image1, reg)[1]

        if nb_samples == 1:
            plt.subplot(len(noises), nb_subs + len(nbs), 1 + j)
            plt.imshow(patterns[0], vmin=0, vmax=1+noises[-1])
            if j == 0:
                plt.title('subject 1')
            plt.ylabel('noise:\n {}'.format(noise), rotation='horizontal')
            plt.subplot(len(noises), nb_subs + len(nbs), 2 + j)
            plt.imshow(patterns[nb_samples], vmin=0, vmax=1+noises[-1])
            if j == 0:
                plt.title('subject 2')


        bary_wass_image1, log = dual.entropic_barycenters(A, M_image1, reg, c=-5,
                                                          max_iter=300, tol=1e-4, log=True)

        print('image min max', patterns[0].min(), patterns[0].max())
        print('histograms min max', A.min(), A.max())
        print('barycenter min max', bary_wass_image1.min(), bary_wass_image1.max())
        plt.subplot(len(noises), nb_subs + len(nbs), nb_samples+ 2 + j)
        # plt.imshow(bary_wass_image1.reshape(patterns[0].shape))
        plt.imshow(bary_wass_image1.reshape(patterns[0].shape), vmin=0, vmax=vmax)
        if j == 0:
            plt.title('barycenter\n{} images\n{}'.format(nb_samples*2, vmax))
    j += nb_subs + len(nbs)



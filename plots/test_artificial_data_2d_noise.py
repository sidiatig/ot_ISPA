import numpy as np
import ot
import matplotlib.pyplot as plt
import os
import sys
main_dir = os.path.split(os.getcwd())[0]
# main_dir = main_dir + '/ot_ISPA'
fig_dir = main_dir  + '/figs/distance_matrix'


# 2D

n = 100  # nb bins
n_y = 50
shape = (n, n_y)
dis_shape = (n*n_y, 2)
# bin positions
x = np.arange(n, dtype=np.float64)
reg = 1/800

noises = [0.001, 0.01, 0.05, 0.1, 0.5]
mean = 0
alpha = 1/dis_shape[1]  # 0<=alpha<=1
weights = np.ones(dis_shape[1]) * alpha

# M with 2d coordinates
nx, ny = shape
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(y,x)
coors = np.vstack((xv.flatten(), yv.flatten())).T
coor = np.empty(coors.shape)
coor[:,0] = coors[:,1]
coor[:,1] = coors[:,0]
M_image = ot.utils.dist(coor)
# different weights for M_image
#  when weight is bigger, value of M_image is smaller,
N = 0.5
M_image1 = M_image / (np.max(M_image) * N)


data_method = 'uniform'
vmax = 0.0005
plt.figure()
fig = plt.gcf()
fig.suptitle(data_method + '  noise', fontsize=14)
j = 0
for noise in noises:
    a1 = np.zeros((n, n_y))
    a2 = np.zeros((n, n_y))
    a1[10:20, 3:8] = 1
    a2[80:90, 12:17] = 1
    # gaussian data
    if data_method == 'gaus':
        a1 = a1 + np.random.normal(mean, noise, (n, n_y))
        a2 = a2 + np.random.normal(mean, noise, (n, n_y))
    # uniform noise:  (b - a) * random_sample(size) + a
    else:
        a1 = a1 + 2 * noise * np.random.random_sample((n, n_y)) - noise
        a2 = a2 + 2 * noise * np.random.random_sample((n, n_y)) - noise


    arti = np.vstack((a1.reshape(1, n*n_y), a2.reshape(1, n*n_y)))
    arti_dis = np.empty(arti.shape)
    for i in range(arti.shape[0]):
        data = arti[i] - np.min(arti[i])
        data /= np.sum(data)
        arti_dis[i] = data
    A = np.transpose(arti_dis)
    # arti = np.transpose(arti)

    plt.subplot(len(noises), 4, 1+j)
    plt.imshow(a1, vmin=0, vmax=1+noises[-1])
    if j == 0:
        plt.title('image 1'.format(noise))
    plt.ylabel('noise:\n {}'.format(noise), rotation='horizontal')
    plt.subplot(len(noises), 4, 2+j)
    plt.imshow(a2, vmin=0, vmax=1+noises[-1])
    if j == 0:
        plt.title('image 2'.format(noise))


    bary_wass_image1, log = ot.bregman.barycenter(A, M_image1, reg, weights,
                                                  numItermax=1000, log=True)
    print('image min max', a1.min(), a1.max())
    print('histograms min max', A.min(), A.max())
    print('barycenter min max', bary_wass_image1.min(), bary_wass_image1.max())
    plt.subplot(len(noises), 4, 3 + j)
    plt.imshow(bary_wass_image1.reshape(a1.shape), vmin=0, vmax=vmax)
    if j == 0:
        plt.title('barycenter\nsame scale\n{}'.format(vmax))
    plt.subplot(len(noises), 4, 4 + j)
    plt.imshow(bary_wass_image1.reshape(a1.shape))
    if j == 0:
        plt.title('barycenter\n')
    j += 4

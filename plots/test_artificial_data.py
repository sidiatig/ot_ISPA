import numpy as np
import scipy.io
import ot

import matplotlib.pyplot as plt


arti_dis = np.empty(arti.shape)
for i in range(arti.shape[0]):
    data = arti[i] - np.min(arti[i])
    data /= np.sum(data)
    arti_dis[i] = data
A = np.transpose(arti_dis)

#
# nx, ny = arti[0].shape
# x = np.linspace(0,1,nx)
# y = np.linspace(0, 1, ny)
# xv, yv = np.meshgrid(x,y)
# coors = np.vstack((xv.flatten(), yv.flatten()))
# M = ot.utils.dist(coors.T)
#

M = ot.utils.dist0(A.shape[0])
# M = np.sqrt(M)
N = 1
M /= np.max(M) * N
alpha = 1/A.shape[1]  # 0<=alpha<=1
weights = np.ones(A.shape[1]) * alpha
# bary_l2 = A.dot(weights)
reg = 1/800
bary_wass, log = ot.bregman.barycenter(A, M, reg, weights,
                                       numItermax=100, log=True)


plt.figure()
plt.subplot(1,4,1)
plt.imshow(a1)
plt.subplot(1,4,2)
plt.imshow(a2)
plt.subplot(1,4,3)
plt.imshow(bary_wass.reshape(a1.shape))
plt.subplot(1,4,4)
plt.imshow(M)
plt.title(N)


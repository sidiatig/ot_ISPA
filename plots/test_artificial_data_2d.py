import numpy as np
import ot
import matplotlib.pyplot as plt
import scipy

# 2D
n = 100  # nb bins
# bin positions
x = np.arange(n, dtype=np.float64)


# a1 = np.ones((n,20)) * 0.001 # m= mean, s= std
# a2 = np.ones((n,20)) * 0.001
# a1[10:20] = 1
# a2[60:70] = 1
# arti = np.vstack((a1.reshape(1,2000), a2.reshape(1,2000)))

# # signal is in the middle of image
# a1 = np.ones((n,20)) * 0.001
# a2 = np.ones((n,20)) * 0.001
# a1[10:20,7:13] = 1
# a2[60:70,7:13] = 1
# arti = np.vstack((a1.reshape(1,2000), a2.reshape(1,2000)))

#signal is not in the middle of image
a1 = np.ones((n,20)) * 0.001
a2 = np.ones((n,20)) * 0.001
a1[10:20,3:9] = 1
a2[80:90,11:17] = 1
arti = np.vstack((a1.reshape(1,2000), a2.reshape(1,2000)))

# # artificial data
# # arti_data = scipy.io.loadmat('/home/qiwang/Downloads/artificial_data.mat')
# arti_data = scipy.io.loadmat('/hpc/crise/wang.q/data/artificial_data.mat')
# arti = arti_data['x']
# # arti = np.vstack((arti[1][:,18:26].reshape((1,480)),arti[42][:,18:26].reshape((1,480))))
# a1 = arti[0]
# a2 = arti[50]
# arti = np.vstack((a1.reshape(1,5400), a2.reshape(1,5400)))


arti_dis = np.empty(arti.shape)
for i in range(arti.shape[0]):
    data = arti[i] - np.min(arti[i])
    data /= np.sum(data)
    arti_dis[i] = data
A = np.transpose(arti_dis)
# arti = np.transpose(arti)
alpha = 1/A.shape[1]  # 0<=alpha<=1
weights = np.ones(A.shape[1]) * alpha
# bary_l2 = A.dot(weights)
reg = 1/800
# M with 2d coordinates
nx, ny = a1.shape
x = np.linspace(0,1,nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(y,x)
coors = np.vstack((xv.flatten(), yv.flatten())).T
coor = np.empty(coors.shape)
coor[:,0] = coors[:,1]
coor[:,1] = coors[:,0]

M_image = ot.utils.dist(coor)
N = 1
# M = np.sqrt(M)
M_image /= np.max(M_image) * N
bary_wass_image, log = ot.bregman.barycenter(A, M_image, reg, weights,
                                       numItermax=100, log=True)

M_vector = ot.utils.dist0(A.shape[0])
# M = np.sqrt(M)
N = 1
M_vector /= np.max(M_vector) * N
bary_wass_vector, log = ot.bregman.barycenter(A, M_vector, reg, weights,
                                       numItermax=100, log=True)

plt.figure()
plt.subplot(1,4,1)
plt.imshow(a1)
plt.title('image 1')
plt.subplot(1,4,2)
plt.imshow(a2)
plt.title('image 2')
plt.subplot(1,4,3)
plt.imshow(bary_wass_vector.reshape(a1.shape))
plt.title('barycenter with \nvector distance matrix')
plt.subplot(1,4,4)
plt.imshow(bary_wass_image.reshape(a1.shape))
plt.title('barycenter with \nimage distance matrix')


# different weights for M_image
M_image = ot.utils.dist(coor)
Ns = [0.1, 0.2, 0.5, 1, 5]
plt.figure()
plt.subplot(1, 7, 1)
plt.imshow(a1)
plt.title('image 1')
plt.subplot(1, 7, 2)
plt.imshow(a2)
plt.title('image 2')
for i in range(len(Ns)):
    N = Ns[i]
    # M = np.sqrt(M)
    M_image1 = M_image / (np.max(M_image) * N)
    bary_wass_image1, log = ot.bregman.barycenter(A, M_image1, reg, weights,
                                                  numItermax=100, log=True)
    plt.subplot(1, 7, i + 3)
    plt.imshow(bary_wass_image1.reshape(a1.shape))
    plt.title('barycenter\n{}'.format(N))


# steps in algorithm
K = np.exp(-M / reg)
K[K<1e-300] = 1e-300
UKv = np.dot(K, np.divide(A.T, np.sum(K, axis=0)).T)
u = (ot.bregman.geometricMean(UKv) / UKv.T).T
# UKv = u * np.dot(K, np.divide(A, np.dot(K, u)))
# u = (u.T * ot.bregman.geometricBar(weights, UKv)).T / UKv
barycenter = ot.bregman.geometricBar(weights, UKv)
plt.figure()
plt.subplot(1,6,1)
plt.imshow(M)
plt.title(N)
plt.subplot(1,6,2)
plt.imshow(K)
plt.subplot(1,6,3)
for i in range(A.shape[1]):
    plt.plot(x, A[:, i])
plt.title('A, 2d')
plt.subplot(1,6,4)
for i in range(UKv.shape[1]):
    plt.plot(x, UKv[:, i])
plt.title('UKv, 2d')
plt.subplot(1,6,5)
plt.plot(x, barycenter)
plt.title('bary_wass, 2d')
plt.subplot(1,6,6)
for i in range(u.shape[1]):
    plt.plot(x, u[:,i])
plt.title('u, 2d')

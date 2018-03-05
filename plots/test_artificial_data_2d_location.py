import numpy as np
import ot
import matplotlib.pyplot as plt
import scipy

# 2D
n = 100  # nb bins
# bin positions
x = np.arange(n, dtype=np.float64)

#signal is not in the middle of image, two signals, each signal in one image
s1 = np.ones((n,50)) * 0.001
s2 = np.ones((n,50)) * 0.001
s1[10:20,3:9] = 1
s2[80:90,11:17] = 1

s3 = np.ones((n,50)) * 0.001
s4 = np.ones((n,50)) * 0.001
s3[10:20,3:9] = 1
s4[40:50,31:37] = 1
# more than one activated zone in one image
d1 = np.ones((n,50)) * 0.001
d2 = np.ones((n,50)) * 0.001
d1[10:20,3:9] = 1
d2[80:90,11:17] = 1
d2[40:50,31:37] = 1

d3 = np.ones((n,50)) * 0.001
d4 = np.ones((n,50)) * 0.001
d3[80:90,11:17] = 1
d4[10:20,3:9] = 1
d4[40:50,31:37] = 1

d5 = np.ones((n,50)) * 0.001
d6 = np.ones((n,50)) * 0.001
d5[40:50,31:37] = 1
d6[80:90,11:17] = 1
d6[10:20,3:9] = 1

d7 = np.ones((n,50)) * 0.001
d8 = np.ones((n,50)) * 0.001
d7[40:50,41:47] = 1
d7[60:70,21:27] = 1
d8[10:20,3:9] = 1
d8[80:90,11:17] = 1



plt.figure()
j = 0
datasets = [(s1,s2), (s3,s4), (d1,d2), (d3,d4),(d5,d6), (d7,d8)]
for (a1,a2) in datasets:
    arti = np.vstack((a1.reshape(1, 5000), a2.reshape(1, 5000)))
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
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(y,x)
    coors = np.vstack((xv.flatten(), yv.flatten())).T
    coor = np.empty(coors.shape)
    coor[:,0] = coors[:,1]
    coor[:,1] = coors[:,0]

    # different weights for M_image
    M_image = ot.utils.dist(coor)
    # Ns = [0.1, 0.2, 0.5, 1, 5]
    Ns = [0.5]

    plt.subplot(len(datasets), 2+len(Ns), 1+j)
    plt.imshow(a1)
    if j == 0:
        plt.title('image 1')
    plt.subplot(len(datasets), 2+len(Ns), 2+j)
    plt.imshow(a2)
    if j == 0:
        plt.title('image 2')
    for i in range(len(Ns)):
        N = Ns[i]
        # M = np.sqrt(M)
        M_image1 = M_image / (np.max(M_image) * N)
        bary_wass_image1, log = ot.bregman.barycenter(A, M_image1, reg, weights,
                                                      numItermax=500, log=True)
        plt.subplot(len(datasets), 2+len(Ns), i + 3 + j)
        plt.imshow(bary_wass_image1.reshape(a1.shape))
        if j == 0:
            plt.title('barycenter\n{}'.format(N))
    j += 2 + len(Ns)

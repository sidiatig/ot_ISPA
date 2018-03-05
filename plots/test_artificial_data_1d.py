import numpy as np
import ot

import matplotlib.pyplot as plt
# 1D gauss
n = 100  # nb bins
# bin positions
x = np.arange(n, dtype=np.float64)
# Gaussian distributions
a1 = np.zeros(n)  # m= mean, s= std
a2 = np.zeros(n)
a1[10:20] = 1
a2[60:70] = 1
a1 = a1 / sum(a1)
a2 = a2 / sum(a2)

# creating matrix A containing all distributions
A = np.vstack((a1, a2)).T
n_distributions = A.shape[1]

# loss matrix + normalization
M = ot.utils.dist0(n)
M /= M.max()

#%% barycenter computation
alpha = 0.2  # 0<=alpha<=1
weights = np.array([1 - alpha, alpha])

# l2bary
bary_l2 = A.dot(weights)

# wasserstein
reg = 1e-3
bary_wass = ot.bregman.barycenter(A, M, reg, weights, numItermax=100)

plt.figure()
plt.clf()
plt.subplot(2, 1, 1)
for i in range(n_distributions):
    plt.plot(x, A[:, i])
plt.title('Distributions')

plt.subplot(2, 1, 2)
plt.plot(x, bary_l2, 'r', label='l2')
plt.plot(x, bary_wass, 'g', label='Wasserstein')
plt.legend()
plt.title('Barycenters')
plt.tight_layout()

# M = ot.utils.dist0(n)
# N = 0.2
# M /= M.max() * N
# K = np.exp(-M / reg)
# K[K<1e-300] = 1e-300
# UKv = u * np.dot(K, np.divide(A, np.dot(K, u)))
# u = (u.T * geometricBar(weights, UKv)).T / UKv
# barycenter = geometricBar(weights, UKv)
# plt.figure()
# plt.subplot(1,6,1)
# plt.imshow(M)
# plt.title(N)
# plt.subplot(1,6,2)
# plt.imshow(K)
# plt.subplot(1,6,3)
# for i in range(n_distributions):
#     plt.plot(x, A[:, i])
# plt.title('A')
# plt.subplot(1,6,4)
# for i in range(n_distributions):
#     plt.plot(x, UKv[:, i])
# plt.title('UKv')
# plt.subplot(1,6,5)
# plt.plot(x, barycenter)
# plt.title('bary_wass')
# plt.subplot(1,6,6)
# for i in range(2):
#     plt.plot(x, u[:,i])
# plt.title('u')


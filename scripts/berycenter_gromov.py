import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as fucs

from sklearn.externals import joblib
import scipy.stats as stats
import numpy as np
import time

import scipy as sp

import scipy.ndimage as spi
from scipy.spatial.distance import cdist
import matplotlib.pylab as pl
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
import ot

import matplotlib.pyplot as plt




def smacof_mds(C, dim, max_iter=3000, eps=1e-9):
    """
    Returns an interpolated point cloud following the dissimilarity matrix C
    using SMACOF multidimensional scaling (MDS) in specific dimensionned
    target space

    Parameters
    ----------
    C : ndarray, shape (ns, ns)
        dissimilarity matrix
    dim : int
          dimension of the targeted space
    max_iter :  int
        Maximum number of iterations of the SMACOF algorithm for a single run
    eps : float
        relative tolerance w.r.t stress to declare converge

    Returns
    -------
    npos : ndarray, shape (R, dim)
           Embedded coordinates of the interpolated point cloud (defined with
           one isometry)
    """

    rng = np.random.RandomState(seed=3)

    mds = manifold.MDS(
        dim,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity='precomputed',
        n_init=1)
    pos = mds.fit(C).embedding_

    nmds = manifold.MDS(
        2,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity="precomputed",
        random_state=rng,
        n_init=1)
    npos = nmds.fit_transform(C, init=pos)

    return npos


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))


main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

# args = sys.argv[1:]
# experiment = args[0]
# nor_method = args[1]
# clf_method = args[2]
# source_id = int(args[3])
# target_id = int(args[4])
# op_function = args[5] # 'lpl1' 'l1l2'

metric_xst = 'null'

experiment = 'fmril'
nor_method = 'no'
clf_method = 'logis'
source_id = 1
target_id = 2
op_function = 'l1l2' # 'lpl1' 'l1l2'

if experiment == 'fmril':
    source = fmril
elif experiment == 'fmrir':
    source = fmrir
else:
    source = meg

result_dir = result_dir + '/{}'.format(experiment)
y_target = source.y_target
subjects = np.array(source.subjects)
x_data = source.x_data_pow if experiment == 'meg' else source.x_data
x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
x = x_indi.copy() if nor_method == 'indi' else x_data.copy()

indices_sub = fucs.split_subjects(subjects)
nb_subs = len(indices_sub)

i = source_id
j = target_id
xs = x[indices_sub[i]]
ys = y_target[indices_sub[i]]
xt = x[indices_sub[j]]

N = 160
A = fmril.x_data[:N]
# A2 = np.empty(N,60,90)
# for i in range(len(A)):
#     A2[i] = A[i].reshape(60,90)
B = np.empty(A.shape)
for i in range(A.shape[0]):
    data = A[i] - np.min(A[i])
    data /= np.sum(data)
    B[i] = data
# B2 = np.empty((N,60,90))
xs = []
for i in range(int(N/40)):
    xs.append(B[i*40:i*40+40])
# for i in range(B.shape[0]):
#     B2[i] = B[i].reshape(60,90)
# B2 = np.transpose(B2)

S = int(N/40)
ns = [len(xs[s]) for s in range(S)]
n_samples = 40
# Cs = [sp.spatial.distance.cdist(xs[s], xs[s]) for s in range(S)]

Cs = [rbf_kernel(xs[s], xs[s], gamma=1/(2*np.mean(cdist(xs[s], xs[s],'euclidean'))**2)) for s in range(S)]
# Cs = [rbf_kernel(xs[s], xs[s]) for s in range(S)]
Cs = [cs / np.median(Cs) for cs in Cs]
#
# xs_ori = []
# for i in range(int(N/40)):
#     xs_ori.append(A[i*40:i*40+40])
# Cs = [rbf_kernel(xs_ori[s], xs_ori[s], gamma=1/(2*np.mean(cdist(xs_ori[s], xs_ori[s],'euclidean'))**2)) for s in range(S)]
# Cs = [cs / np.median(Cs) for cs in Cs]
# #
# C = np.empty(A.shape)
# for i in range(A.shape[0]):
#     data = A[i] - np.min(A[i])
#     data /= np.sum(data)
#     C[i] = data
# xs_ori2 = []
# for i in range(int(N/40)):
#     xs_ori2.append(C[i*40:i*40+40])
# Cs = [rbf_kernel(xs_ori2[s], xs_ori2[s], gamma=1/(2*np.mean(cdist(xs_ori2[s], xs_ori2[s],'euclidean'))**2)) for s in range(S)]
# Cs = [cs / np.median(Cs) for cs in Cs]


ps = [ot.unif(ns[s]) for s in range(S)]
p = ot.unif(n_samples)

alpha = 1./S  # 0<=alpha<=1
weights = np.ones(S) * alpha

bary_gromov= ot.gromov.gromov_barycenters(n_samples, Cs,
                                           ps, p, weights, 'square_loss', 5e-3,
                                           max_iter=1000, tol=1e-5, log=False)
print(bary_gromov)
# test = smacof_mds(bary_gromov, 5400, max_iter=3000, eps=1e-9)
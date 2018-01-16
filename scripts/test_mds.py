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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
import ot


x = fmril.x_indi
ids = fucs.split_subjects(fmril.subjects)

for source in range(5):
    for target in range(5):
        if source != target:

            data1 = x[ids[source]]
            y1 = fmril.y_target[ids[source]]
            data2 = x[ids[target]]
            y2 = fmril.y_target[ids[target]]
            C = ot.utils.dist(data1, data1)

            max_iter = 3000
            eps = 1e-9
            dim = 5400

            rng = np.random.RandomState(seed=3)

            mds = manifold.MDS(
                dim,
                max_iter=max_iter,
                eps=eps,
                dissimilarity='precomputed',
                n_init=1)
            pos = mds.fit(C).embedding_

            nmds = manifold.MDS(
                dim,
                max_iter=max_iter,
                eps=eps,
                dissimilarity="precomputed",
                random_state=rng,
                n_init=1)
            npos = nmds.fit_transform(C, init=pos)

            npos_indi = StandardScaler().fit_transform(npos)

            _, _, mds_acc, _ = fucs.gridsearch_persub(npos_indi, y1, data2, y2, clf_method='svm',njobs=3)
            _, _, indi_acc, _ = fucs.gridsearch_persub(data1, y1, data2, y2, clf_method='svm',njobs=3)
            print(source, target, mds_acc, indi_acc)



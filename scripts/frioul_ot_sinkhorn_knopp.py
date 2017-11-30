from .data import fmri_data_cv as fmril
from .data import fmri_data_cv_rh as fmrir
from .data import meg_data_cv as meg
from .model import procedure_function as models
# import fmri_data_cv as fmril
# import fmri_data_cv_rh as fmrir
# import meg_data_cv as meg
# import procedure_function as models

import numpy as np
import time
import os
import sys

from scipy.spatial.distance import cdist

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

args = sys.argv[1:]
experiment = args[0]
nor_method = args[1]
clf_method = args[2]
cv_start = int(args[3])

note = '{}_{}_{}_{}cvstart'.format(experiment, nor_method, clf_method,
                                          cv_start)
print('optimal skinkhorn_knopp', note)

if experiment == 'fmril':
    source = fmril
elif experiment == 'fmrir':
    source = fmrir
else:
    source = meg

y_target = source.y_target
subjects = np.array(source.subjects)
nb_samples_each = source.nb_samples_each
x_data = source.x_data_pow if experiment == 'meg' else source.x_data
x_indi = source.x_indi_pow if experiment == 'meg' else source.x_indi
x = x_indi if nor_method == 'indi' else x_data
cross_v = source.cross_v[cv_start]

x, y_target = models.shuffle_data(x, y_target, nb_samples_each)
reg = 1e-1
cpt = 0
err = 1
numItermax=1000
stopThr=1e-9
verbose=False
log=False

x_ot = np.empty(x.shape)
Xt = x[:nb_samples_each]
x_ot[:nb_samples_each] = Xt
for i in range(1, int(x.shape[0]/nb_samples_each)):
    start = nb_samples_each * i
    end = nb_samples_each * (i+1)
    Xs = x[start:end]
    M = cdist(Xs, Xt, metric='euclidean')
    if M.max() >= 20:
        M = np.sqrt(M)
    M = np.asarray(M, dtype=np.float64)
    a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    K = np.exp(-M / reg)
    Kp = (1 / a).reshape(-1, 1) * K

    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + \
                      np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b)) ** 2
            # if log:
            #     log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1

    coupling = u.reshape((-1, 1)) * K * v.reshape((1, -1))

    transp = coupling / np.sum(coupling, 1)[:, None]
    # set nans to 0
    transp[~ np.isfinite(transp)] = 0
    # compute transported samples
    transp_Xs = np.dot(transp, Xt)
    x_ot[start:end] = transp_Xs

result_ot = models.gridsearch(x_ot,y_target,subjects,cross_v,
                              experiment, clf_method,nor_method,cv_start,njobs=3)
np.save(result_dir+'/ot_sinkhorn_knopp_{}'.format(note),result_ot)

print('optimal skinkhorn_knopp', note)
print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

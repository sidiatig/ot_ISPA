import data.fmri_data_cv as fmril
import data.fmri_data_cv_rh as fmrir
import data.meg_data_cv as meg
import model.procedure_function as models

import numpy as np
import time
import os
import ot

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))
main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

experiment = 'fmril'
nor_method = 'indi'
clf_method = 'logis'
cv_start = 0

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
x_ot = np.empty(x.shape)
Xt = x[:nb_samples_each]
x_ot[:nb_samples_each] = Xt
for i in range(1, int(x.shape[0]/nb_samples_each)):
    start = nb_samples_each * i
    end = nb_samples_each * (i+1)
    Xs = x[start:end]
    M = ot.dist(Xs, Xt, metric='euclidean')
    if M.max() >= 20:
        M = np.sqrt(M)
    M = np.asarray(M, dtype=np.float64)
    a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]
    coupling = ot.bregman.sinkhorn_knopp(a, b, M, reg)
    transp = coupling / np.sum(coupling, 1)[:, None]
    # set nans to 0
    transp[~ np.isfinite(transp)] = 0
    # compute transported samples
    transp_Xs = np.dot(transp, Xt)
    x_ot[start:end] = transp_Xs

result_ot = models.gridsearch(x_ot,y_target,subjects,cross_v,
                              experiment, clf_method,nor_method,cv_start)
np.save(result_dir+'/ot_sinkhorn_knopp_{}'.format(note),result_ot)

print('optimal skinkhorn_knopp', note)
print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

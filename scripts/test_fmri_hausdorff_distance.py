import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as models
import ot

from sklearn.externals import joblib
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import time

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

args = sys.argv[1:]
experiment = 'fmril'
nor_method = 'indi'
clf_method = 'logis'
source_id = 1
target_id = 2
op_function = 'l1l2'
note = '{}s_{}t_{}_{}_{}_skinkhorn_{}'.format(source_id, target_id,
                                                   experiment, nor_method, clf_method,
                                                 op_function)
print(note)

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

i = 1
j = 2
indices_sub = models.split_subjects(subjects)
xs = x[indices_sub[i]]
ys = y_target[indices_sub[i]]
xt = x[indices_sub[j]]
yt = y_target[indices_sub[j]]

regs = [1e-3,1e-2, 1e-1, 1, 10]
etas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
trans_xs = [[] for i in range(len(regs))]
reg_nb = 0
for reg in regs:
    for eta in etas:
        print(reg, eta)
        ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta)
        ot_l1l2.fit(Xs=xs, Xt=xt, ys=ys)
        xst = ot_l1l2.transform(Xs=xs)
        dicrec1 = directed_hausdorff(xst, xt)[0]
        dicrec2 = directed_hausdorff(xt, xst)[0]
        print('general', max(dicrec1, dicrec2))
        trans_xs[reg_nb].append(max(dicrec1,dicrec2))
    reg_nb += 1

ori_best, ori_grid, ori_score = models.gridsearch_persub(xs,ys,xt,yt)
print(ori_score, ori_best)
ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=0.001, reg_cl=0.001)
ot_l1l2.fit(Xs=xs, Xt=xt, ys=ys)
xst = ot_l1l2.transform(Xs=xs)
ot_best, ot_grid, ot_score = models.gridsearch_persub(xst,ys,xt,yt)
print(ot_best,ot_score)

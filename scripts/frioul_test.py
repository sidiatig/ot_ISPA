import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg
from model import procedure_function as models
import opt

from sklearn.externals import joblib
import numpy as np
import time


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

args = sys.argv[1:]
experiment = args[0]
nor_method = args[1]
clf_method = args[2]
source_id = int(args[3])
op_function = args[4] # 'lpl1' 'l1l2'
exp_dir = result_dir +'/{}'.format(experiment)
# args = sys.argv[1:]
# experiment = 'fmril'
# nor_method = 'indi'
# clf_method = 'logis'
# source_id = 2
# target_id = 1


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

indices_sub = models.split_subjects(subjects)
nb_subs = len(indices_sub)

# for i in range(len(indices_sub)-1):
#     for j in range(i+1, len(indices_sub)):
i = source_id
for j in range(nb_subs):
    if i != j:
        scores_params = {'source': [], 'target': [], 'ori_score': [], 'ot_score': [],
                         'params': []}
        note = '{}s_{}t_{}_{}_{}_skinkhorn_{}'.format(i, j,
                                                      experiment, nor_method, clf_method, op_function)

        print(note)
        xs = x[indices_sub[i]]
        ys = y_target[indices_sub[i]]
        xt = x[indices_sub[j]]
        yt = y_target[indices_sub[j]]
        M = opt.utils.dist(xs,xt)
        M = opt.utils.cost_normalization(M, 'max')
        print('M max',M.max())
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]

        ori_best, ori_grid, ori_score = models.gridsearch_persub(xs,ys,xt,yt,experiment,clf_method)
        print('original source data predicts on original target, accuracy', ori_score)
        print('original source data best params', ori_best)

        scores_params['source'].append(i)
        scores_params['target'].append(j)
        scores_params['ori_score'].append(ori_score)

        ot_scores = []
        ot_params = {}

        # reg can't be too small, otherwize K will be 0
        # when reg is larger, , prediction score is better,
        for reg in [1e-3,1e-2,1e-1,1,10,100,1000]:
            for eta in [1e-4,1e-3,1e-2,1e-1,1,1e1]:
                if op_function == 'lpl1':
                    G_lpl1 = opt.da.sinkhorn_lpl1_mm(a=a, labels_a=ys, b=b, M=M, reg = reg, eta = eta)
                    xst = opt.da.transform(G_lpl1, xt)
                elif op_function == 'l1l2':
                    G_l1l2 = opt.da.sinkhorn_l1l2_gl(a=a, labels_a=ys, b=b, M=M, reg = reg, eta = eta)
                    xst = opt.da.transform(G_l1l2, xt)



                best_param, grid_cvs, score_ot = models.gridsearch_persub(xst,ys, xt,yt, experiment,clf_method)
                print('reg:{}, eta:{}, best_param:{}'.format(reg,eta,best_param))
                print('score_ot', score_ot)
                if len(ot_scores)==0:
                    ot_scores.append(score_ot)
                    ot_params['reg'] = reg
                    ot_params['eta'] = eta
                    ot_params['C'] = best_param['C']
                    ot_params['penalty'] = best_param['penalty']
                    ot_params['score_ot'] = score_ot
                else:
                    if score_ot > max(ot_scores):
                        ot_params['reg'] = reg
                        ot_params['eta'] = eta
                        ot_params['C'] = best_param['C']
                        ot_params['penalty'] = best_param['penalty']
                        ot_params['score_ot'] = score_ot
                    ot_scores.append(score_ot)

        scores_params['ot_score'].append(ot_params['score_ot'])
        scores_params['params'].append(ot_params)
        print('final params', scores_params)
        joblib.dump(scores_params, exp_dir + '/{}_score_params.pkl'.format(note))
        if scores_params['ot_score'] <= scores_params['ori_score']:
            joblib.dump(scores_params, exp_dir+'/gridtables/{}_small_ot.pkl'.format(note))


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))







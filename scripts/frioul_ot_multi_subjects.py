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
cv_start = int(args[4])
op_function = 'l1l2' # 'lpl1' 'l1l2'
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




import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_data_cv as fmril
from data import fmri_data_cv_rh as fmrir
from data import meg_data_cv as meg

import scipy.io

main_dir = os.path.split(os.getcwd())[0]
scipy.io.savemat(main_dir + '/fmri_left.mat',
                 dict(x=fmril.x_data,y=fmril.y_target,subjects=fmril.subjects))
scipy.io.savemat(main_dir + '/fmri_right.mat',
                 dict(x=fmrir.x_data,y=fmrir.y_target,subjects=fmrir.subjects))



import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

from data import fmri_localizer as newfmri
from model import utils
import numpy as np

x = newfmri.x_indi
y = newfmri.y_target
subs = newfmri.subjects

cv = utils.create_cv(x, y, subs, 4)
cv = np.asarray(cv)
ids = np.arange(len(cv))
np.random.shuffle(ids)
cv = cv[ids]
np.save(main_dir + '/data/cv_newfmri.npy', cv)



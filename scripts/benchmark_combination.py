import numpy as np
import os
from sklearn.externals import joblib
from scipy.stats import ttest_rel

main_dir = os.path.split(os.getcwd())[0]
data_dir = main_dir + '/results/newfmri/bench'
indi = []
no = []
for i in range(100):

    data = joblib.load(data_dir + '/newfmri_indi_logis_split{}_banchmark.pkl'.format(i))
    indi.append(data['score'])
    data = joblib.load(data_dir + '/newfmri_no_logis_split{}_banchmark.pkl'.format(i))
    no.append(data['score'])
print(indi)
print(no)
print(len(indi), len(no))
print(ttest_rel(indi,no))
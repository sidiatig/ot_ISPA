import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ot
data_dir = '/hpc/crise/wang.q/data/localizer_betas_vs_trial_type'
x_data = joblib.load(data_dir + '/localizer_betas_temporal.jl')
label6 = joblib.load(data_dir + '/localizer_trialtypes_6labels.jl')
label2 = joblib.load(data_dir + '/localizer_trialtypes_2labels.jl')
subjects = joblib.load(data_dir + '/subjects_id.jl')
y_target = np.array([1 if i == 'voice' else 0 for i in label2])

index_subjects = []
for subject in np.unique(subjects):
    index = [i for i, j in enumerate(subjects) if j==subject]
    index_subjects.append(index)

x_indi = np.empty(x_data.shape)
for j in range(len(index_subjects)):
    x_indi[index_subjects[j]] = StandardScaler().fit_transform(x_data[index_subjects[j]])

x1 = x_indi[index_subjects[1]]
x2 = x_indi[index_subjects[2]]
pca = PCA(n_components=100)
x1_pca = pca.fit_transform(x1)
y1 = y_target[index_subjects[1]]
y2 = y_target[index_subjects[2]]
x2_pca = pca.fit_transform(x2)

reg = 1
eta = 0.1
ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, norm='max')
ot_l1l2.fit(Xs=x1_pca, Xt=x2_pca, ys=y1)
xst = ot_l1l2.transform(Xs=x1_pca)

print(xst)
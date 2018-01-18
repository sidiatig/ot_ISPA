
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

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
zero_col = []
for i in range(x_indi.shape[1]):
    if np.all(x_indi[:,i]==0):
        zero_col.append(i)
zero_col = np.array(zero_col)
x_indi = np.delete(x_indi, zero_col,1)


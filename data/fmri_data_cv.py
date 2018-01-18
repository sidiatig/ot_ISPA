import os.path as op
import tables
import numpy as np
from sklearn.preprocessing import StandardScaler

hemi = 'lh'        # data from left hemisphere or right hemisphere
method = 'transfer'  # 'transfer', 'multi' or 'multi2', different cross validation methods
clf_name = 'logis' # 'logis' for logistic regresion, 'svm' for svm
nor_name = 'stand' # 'stand' for StandardScaler
split_nb = 1000    # number of the cross validation splits
nb_samples_each = 40

try:
    result_dir = '/home/qiwang/Downloads'
    proj_dir = result_dir+ '/wang.q/data/100_subjects_voice_classification/'
    # get data
    data_dir = proj_dir + 'data/'
    data_path = op.join(data_dir, '{}.100subjects_data.h5'.format(hemi))
    h5file = tables.open_file(data_path, driver='H5FD_CORE')
except:
    result_dir = '/hpc/crise'
    proj_dir = result_dir + '/wang.q/data/100_subjects_voice_classification/'
    # get data
    data_dir = proj_dir + 'data/'
    data_path = op.join(data_dir, '{}.100subjects_data.h5'.format(hemi))
    h5file = tables.open_file(data_path, driver='H5FD_CORE')
data = h5file.root.data[:]
y_class = h5file.root.y_class[:]
# 'CAT16','CBY19'
subjects = h5file.root.subjects[:]
h5file.close()
n_samples, n_x, n_y = data.shape
x_data = data.reshape(n_samples, n_x * n_y)
y_target = np.array([1 if i == b'VO' else 0 for i in y_class])
new_subjects = []
for i,subject in enumerate(np.unique(subjects)):
    index = [i+1 for j in subjects if j==subject]
    new_subjects += index
subjects = np.array(new_subjects)

# index_sujbects:[[0,1,...39],[40,...79]...]
index_subjects = []
for subject in np.unique(subjects):
    index = [i for i, j in enumerate(subjects) if j==subject]
    index_subjects.append(index)

x_indi = np.empty(x_data.shape)
for j in range(int(x_data.shape[0]/nb_samples_each)):
    start = j * nb_samples_each
    end = start + nb_samples_each
    x_indi[start:end] = StandardScaler().fit_transform(x_data[start:end])
# abstract cross validation splits
cv_name = 'data/{0}_{1}splits.npy'.format(method, split_nb)
cross_val = np.load(proj_dir + cv_name)
cross_v = []
for train_index, test_index in cross_val:
    cross_v.append((train_index, test_index))


import os.path as op
import tables
import numpy as np
from sklearn.preprocessing import StandardScaler

hemi = 'whole'  # we don't have left or right hemisphere data
method = 'transfer'  # 'transfer', 'multi' or 'multi2', different cross validation methods
clf_name = 'logis'  # 'logis' for logistic regresion, 'svm' for svm
nor_name = 'stand'
split_nb = 120
nb_samples_each = 576

try:
    result_dir = '/home/qiwang/Downloads'
    proj_dir = result_dir+ '/wang.q/data/meg_decode/'
    # get data
    data_dir = proj_dir + 'data/'
    data_path = op.join(data_dir, '16subjects_meg_data_aranged.h5')
    h5file = tables.open_file(data_path, driver='H5FD_CORE')
except:
    result_dir = '/hpc/crise'
    proj_dir = result_dir+ '/wang.q/data/meg_decode/'
    # get data
    data_dir = proj_dir + 'data/'
    data_path = op.join(data_dir, '16subjects_meg_data_aranged.h5')
    h5file = tables.open_file(data_path, driver='H5FD_CORE')

x_data = h5file.root.X_train[:]
y_target = h5file.root.y_train[:]
subjects = h5file.root.s_train[:]
h5file.close()
#subjects:[1,1,1,...,2,2,2...,16,16,16]
subjects = np.array(subjects)
# index_subjects: [[0,1,2,...575], [576,577,...]...]
index_subjects = []
for subject in np.unique(subjects):
    index = [i for i, j in enumerate(subjects) if j==subject]
    index_subjects.append(index)
x_data_pow = x_data * pow(10,12)
x_indi = np.empty(x_data.shape)
for j in range(int(x_data.shape[0]/nb_samples_each)):
    start = j * nb_samples_each
    end = start + nb_samples_each
    x_indi[start:end] = StandardScaler().fit_transform(x_data[start:end])
x_indi_pow = np.empty(x_data_pow.shape)
for j in range(int(x_data_pow.shape[0]/nb_samples_each)):
    start = j * nb_samples_each
    end = start + nb_samples_each
    x_indi_pow[start:end] = StandardScaler().fit_transform(x_data_pow[start:end])

# cross validation
cv_name = 'data/{0}_{1}splits_aranged.npy'.format(method, split_nb)
cross_val = np.load(proj_dir + cv_name)
cross_v = []
for train_index, test_index in cross_val:
    cross_v.append((train_index, test_index))
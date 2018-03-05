import numpy as np

# size of the image
x_size = 60
y_size = 90

# definition of activated rectangle in each subject
x_min_list = [10,40]
x_max_list = [20,50]
y_min_list = [20,20]
y_max_list = [25,25]

# number of subjects
n_subj = len(x_min_list)

# number of samples per subject
n_samples = 40

# create true activation patterns
# np.random.random is the continuous uniform distribution over [0.0,1.0)
noisefree_patterns = np.zeros([n_subj,x_size,y_size])
all_patterns = np.zeros([n_subj*n_samples,x_size,y_size])
for subj_idx in range(n_subj):
    pattern = np.zeros([x_size,y_size])
    pattern[x_min_list[subj_idx]:x_max_list[subj_idx],y_min_list[subj_idx]:y_max_list[subj_idx]] = 1
    noisefree_patterns[subj_idx] = pattern
    for sample_idx in range(n_samples):
        all_patterns[sample_idx + subj_idx*n_samples] = pattern + np.random.random(size=[x_size,y_size])
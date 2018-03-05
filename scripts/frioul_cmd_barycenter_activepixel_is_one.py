from os import system
import numpy as np

code_dir = '/hpc/crise/wang.q/src_code/Pycharm_Projects/ot_ISPA/scripts'

noise_type = 'gaus'
# regs = [0.01, 0.003, 0.001, 0.0008, 0.0005]  # 0.001 is the best
# noises = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# 'fixed', 2subs, 1zones,
# 'dif', 2subs, 1zones,
# 'dif', 1subs, 1zones,
# 'dif', 1subs, 2zones,

regs = [0.001]
cs = [-4] # step for gradient
# noises = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1, 0.5]
noises = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1, 0.5]
nbs_samples = [i for i in range(19,40,3)]   # n_samples is the nb of samples in each subject
nb_subs = 1
nb_zones = 2
Ns = [0.5]  # N is the weight of distance matrix
loc_type = 'dif'  # 'fixed' or 'dif'
bary_type = 'bregman'


for reg in regs:
    for c in cs:
        for noise in noises:
            for N in Ns:
                for nb_samples in nbs_samples:
                    cmd = "frioul_batch -n '11,12,13,14,15,16,17' -c 3 " \
                          "'/hpc/crise/anaconda3/bin/python3.5 " \
                          "%s/frioul_barycenter_activepixel_is_gaus.py %s %f %f %f %d %d %d %f %s %s'" \
                          % (code_dir, noise_type, reg, c, noise, nb_samples,
                             nb_subs, nb_zones, N, loc_type, bary_type)
                    # a = commands.getoutput(cmd)
                    a = system(cmd)
                    print(cmd)

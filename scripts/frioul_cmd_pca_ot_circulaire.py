# pca + optimal transport + circulaire, several source one target

# learn d components from source by pca,
# transform them on target
# use circulaire to learn reg, eta for optimal transform
# learn coupling by learned reg eta
# transform target with weighted source
from os import system
import os
import itertools
import numpy as np

dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/scripts'


experiment = 'fmril'
nor_method = 'indi'
clf_method = 'logis'
ot_method = 'l1l2'

target_subs = [i for i in range(10)]

# id = 0
# sources = [1, 2, 3, 4, 5, 6, 7, 8]
# cmd = "frioul_batch -n '11,12,13,14,15,16,17' -c 3 " \
#       "'/hpc/crise/anaconda3/bin/python3.5 " \
#       "%s/frioul_pca_ot_circulaire_newdata.py %s %s %s %s %s %d'" \
#       % (code_dir, experiment, nor_method, clf_method, ot_method, sources, target_subs[id])
# # a = commands.getoutput(cmd)
# a = system(cmd)
# print(cmd)



for id in range(len(target_subs)):
    source_subs = np.delete(target_subs, id)
    for sources in list(itertools.combinations(source_subs, len(source_subs)-1)):
        sources = list(sources)
        cmd = "frioul_batch -n '11,12,13,14,15,16,17' -c 3 " \
              "'/hpc/crise/anaconda3/bin/python3.5 " \
              "%s/frioul_pca_ot_circulaire_fmril.py %s %s %s %s %s %d'" \
              % (code_dir, experiment, nor_method, clf_method, ot_method, sources, target_subs[id])
        # a = commands.getoutput(cmd)
        a = system(cmd)
        print(cmd)
    #     break
    # break

from os import system
import os

dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/scripts'


experiments = ['fmril']
nor_methods = ['no'] #  'no'
clf_methods = ['svm']
source_ids = [i for i in range(100)]
target_ids = [i for i in range(5)]
op_function = 'maplin'

for experiment in experiments:
    for nor_method in nor_methods:
        for clf_method in clf_methods:
            for source_id in source_ids:
                for target_id in target_ids:
                    if source_id != target_id:
                        cmd = "frioul_batch -n '11,12,13,14,15,16,17,18,19,20' -c 3 " \
                              "'/hpc/crise/anaconda3/bin/python3.5 " \
                              "%s/frioul_ot_pair_circulaire.py %s %s %s %d %d %s'" \
                              % (code_dir, experiment, nor_method, clf_method, source_id, target_id, op_function)
                        # a = commands.getoutput(cmd)
                        a = system(cmd)
                        print(cmd)

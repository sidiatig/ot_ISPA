from os import system
import os

dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/scripts'


experiments = ['newfmri']
nor_methods = ['no'] #  'no'
clf_methods = ['logis']
# nb_splits = [i for i in range(50,100)]
nb_splits = [3, 8, 29]
for experiment in experiments:
    for nor_method in nor_methods:
        for clf_method in clf_methods:
            for nb_split in nb_splits:
                cmd = "frioul_batch -n '11,12,13,14,15,16,17,18' -c 4 " \
                      "'/hpc/crise/anaconda3/bin/python3.5 " \
                      "%s/frioul_benchmark_newfmri.py %s %s %s %d'" \
                      % (code_dir, experiment, nor_method, clf_method, nb_split)
                # a = commands.getoutput(cmd)
                a = system(cmd)
                print(cmd)

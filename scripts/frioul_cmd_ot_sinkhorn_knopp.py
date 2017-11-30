from os import system
import os


dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/script'

experiments = ['meg']
data_methods = ['indi'] #  'no'
clf_methods = ['logis']
cv_starts = [i for i in range(120)]



for experiment in experiments:
    for data_method in data_methods:
        for clf_method in clf_methods:
            for cv_start in cv_starts:
                cmd = "frioul_batch -n '11,12,13,16,17,18' -c 3 " \
                      "'/hpc/crise/anaconda3/bin/python3.5 " \
                      "%s/frioul_ot_sinkhorn_knopp.py %s %s %s %d'" \
                      % (code_dir, experiment, data_method, clf_method, cv_start)

                # a = commands.getoutput(cmd)
                a = system(cmd)
                print(cmd)

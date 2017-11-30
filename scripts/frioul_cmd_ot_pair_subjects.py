from os import system
import os

dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/script'

experiments = ['fmril']
data_methods = ['indi'] #  'no'
clf_methods = ['logis']
nb_subs = 100

for experiment in experiments:
    for data_method in data_methods:
        for clf_method in clf_methods:
            for source_id in range(nb_subs):
                cmd = "frioul_batch -n '11,12,13,14,15,16,17,18' -c 3 " \
                      "'/hpc/crise/anaconda3/bin/python3.5 " \
                      "%s/frioul_test.py %s %s %s %d'" \
                      % (code_dir, experiment, data_method, clf_method, source_id)
                # a = commands.getoutput(cmd)
                a = system(cmd)
                print(cmd)

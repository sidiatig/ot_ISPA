from os import system
import os
import itertools
dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/scripts'


experiment = 'fmril'
nor_method = 'indi'
dist_type = 'matrix'
reg = 1/1000
ids = list(itertools.combinations([i for i in range(20)],2))
ite = 500

for id in ids:
    cmd = "frioul_batch -n '11,12,13,14,15,16,17' -c 3 " \
          "'/hpc/crise/anaconda3/bin/python3.5 " \
          "%s/frioul_barycenter_nsubs.py %s %s %s %d %d'" \
          % (code_dir, experiment, nor_method, dist_type, id[0], id[1])
    # a = commands.getoutput(cmd)
    a = system(cmd)
    print(cmd)

#
# list(itertools.combinations([1,2,3,4],2))
from os import system
import os

dir = os.path.split(os.getcwd())[0]
code_dir = dir + '/scripts'


reg = 0.01
ids = [i for i in range(50)] # step for gradient
experiment = 'fmril'
datamethod = 'indi'

for id in ids:
    cmd = "frioul_batch -n '11,12,13,14,15,16,17' -c 3 " \
          "'/hpc/crise/anaconda3/bin/python3.5 " \
          "%s/frioul_barycenter_clf.py %f %d %s %s'" \
          % (code_dir, reg, id, experiment, datamethod)
    # a = commands.getoutput(cmd)
    a = system(cmd)
    print(cmd)


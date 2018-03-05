from os import system


for id in range(895936,895999):
    cmd = "frioul_del_jobs -i %d" % (id)
    # a = commands.getoutput(cmd)
    a = system(cmd)
    print(cmd)
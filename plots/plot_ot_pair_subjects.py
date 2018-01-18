import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

import matplotlib
matplotlib.use('Agg')

from sklearn.externals import joblib
import matplotlib.pylab as plt
import numpy as np
import time


print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))

main_dir = os.path.split(os.getcwd())[0]
data_dir = main_dir + '/results'
for i in range(65):
    for j in range(5):
        print('s,t', i,j)
        note = 'h_{}s_{}t_fmril_no_svm_l1l2_pair_params'.format(i, j)
        try:
            joblib.load(data_dir + '/fmril/big_base/{}.pkl'.format(note))
        except:
            print()
            # print('no such file, s,t', i,j)
        else:
            data = joblib.load(data_dir + '/fmril/big_base/{}.pkl'.format(note))
            params_acc = data['params_acc']
            reg_coor = np.log10(params_acc['params'])
            zero_coor = reg_coor[params_acc['acc'] == 0]
            other_coor = reg_coor[params_acc['acc'] != 0]
            plt.figure(figsize=(9, 5))
            plt.subplot(1, 2, 1)
            plt.plot(zero_coor[:, 0], zero_coor[:, 1], '+r', label='acc is 0.5')
            plt.plot(other_coor[:, 0], other_coor[:, 1], 'ob', label='acc not 0.5')
            plt.legend(loc=0)
            plt.title(note)

            plt.subplot(1, 2, 2)
            accs = params_acc['acc'].reshape((6, 6))
            accs2 = np.empty((accs.T.shape))
            for k in range(accs.shape[0]):
                accs2[:, k] = accs[k][::-1]

            plt.imshow(accs2)
            plt.colorbar()
            plt.title('imshow')
            plt.savefig(main_dir + '/figs/acc_imshow_{}.png'.format(note))

print(time.strftime('%Y-%m-%d %A %X %Z', time.localtime(time.time())))







import os
import sys

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'
sys.path.append(main_dir)

import matplotlib.pylab as plt
import numpy as np


A = np.arange(10).reshape((5,2))
plt.figure()
plt.scatter(A[:, 0], A[:, 1], '+r', label='acc is 0.5')

print('2')
plt.legend(loc=0)
plt.title('test')
plt.savefig(main_dir + '/figs/{}.png'.format('test'))

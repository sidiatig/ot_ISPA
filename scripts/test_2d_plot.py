import ot
import matplotlib.pylab as pl
import numpy as np
from scipy.spatial.distance import directed_hausdorff
n_source_samples = 150
n_target_samples = 150

xs, ys = ot.datasets.get_data_classif('3gauss', n_source_samples)
xt, yt = ot.datasets.get_data_classif('3gauss2', n_target_samples)

M = ot.utils.dist(xs,xt)
M = M/ ot.utils.cost_normalization(M,'max')
a = np.ones((xs.shape[0]))/xs.shape[0]
b = np.ones((xt.shape[0]))/xt.shape[0]

trans_xs = []
# G = ot.da.sinkhorn_l1l2_gl(a=a, labels_a=ys,b=b,M=M,reg=1e-2,)
# xst = len(a) * np.dot(G,xt)
regs = [1e-2, 1e-1, 1, 10, 100, 1000]
eta = 1e-2
for reg in regs:
    print(reg)
    ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=eta, max_iter=20)

    ot_l1l2.fit(Xs=xs, Xt=xt, ys=ys)
    xst = ot_l1l2.transform(Xs=xs)
    trans_xs.append(xst)


pl.figure()
pl.subplot(1,8,1)
pl.scatter(xs[:,0],xs[:,1], marker='o',c=ys, label='source data')

pl.subplot(1,8,2)
pl.scatter(xt[:,0],xt[:,1], marker='*', c=yt, label='target data')

for i in range(3,9):
    pl.subplot(1,8,i)
    xst = trans_xs[i-3]
    pl.scatter(xst[:,0],xst[:,1], marker='+',c=ys)
    pl.title('l1l2, {}'.format(regs[i-3]))


for j in range(len(trans_xs)):
    print(j)
    print('reg:{},eta:{}'.format(regs[j],eta))
    dicrec1 = directed_hausdorff(trans_xs[j],xt)[0]
    dicrec2 = directed_hausdorff(xt, trans_xs[j])[0]
    print(dicrec1)
    print(dicrec2)
    print('general', max(dicrec1,dicrec2))



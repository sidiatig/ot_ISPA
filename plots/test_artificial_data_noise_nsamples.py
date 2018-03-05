import numpy as np
import ot
import matplotlib.pyplot as plt

# 2 subjects, each subject has one zone,
# uniform noise:  (b - a) * random_sample(size) + a
# noise is uniformly distributed in the interval [a,b)
# gaussian noise: np.random.normal(mu, sigma, size)

x_min = [10, 70]
x_max = [20, 80]
y_min = [5, 40]
y_max = [10, 45]

x_size = 100
y_size = 50

Ns = 0.5
noises = [0.001, 0.01, 0.05, 0.1, 0.5]
reg = 1 / 800
nb_subs = 2

# M with 2d coordinates
nx, ny = x_size, y_size
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(y,x)
coors = np.vstack((xv.flatten(), yv.flatten())).T
coor = np.empty(coors.shape)
coor[:,0] = coors[:,1]
coor[:,1] = coors[:,0]
M_image = ot.utils.dist(coor)


noisefrees = np.zeros([nb_subs, x_size, y_size])
for sub_id in range(nb_subs):
    noisefrees[sub_id][x_min[sub_id]:x_max[sub_id], y_min[sub_id]:y_max[sub_id]] = 1

plt.figure()
plt.subplot(1,2,1)
plt.imshow(noisefrees[0])
plt.subplot(1,2,2)
plt.imshow(noisefrees[1])


plt.figure()
fig = plt.gcf()
fig.suptitle('two subjects with several samples', fontsize=14)
j = 0
nbs = [i for i in range(1,6)]
for noise in noises:
    # nb_samples is nb of samples in each sub
    # (b - a) * random_sample() + a is the continuous uniform distribution over [a,b)
    for nb_samples in nbs:
        alpha = 1 / (nb_subs * nb_samples)  # 0<=alpha<=1
        weights = np.ones(nb_subs * nb_samples) * alpha

        patterns = np.zeros([nb_subs * nb_samples, x_size, y_size])
        for sub_id in range(nb_subs):
            for sample_id in range(nb_samples):
                patterns[sample_id + sub_id * nb_samples] = noisefrees[sub_id] + \
                                                            noise * np.random.random_sample((x_size, y_size))


        arti = patterns.reshape([-1, x_size*y_size])
        arti_dis = np.empty(arti.shape)
        for i in range(arti.shape[0]):
            data = arti[i] - np.min(arti[i])
            data /= np.sum(data)
            arti_dis[i] = data
        A = np.transpose(arti_dis)

        if nb_samples == 1:
            plt.subplot(len(noises), nb_subs + len(nbs), 1 + j)
            plt.imshow(patterns[0], vmin=0, vmax=1+noises[-1])
            if j == 0:
                plt.title('subject 1')
            plt.ylabel('noise:\n {}'.format(noise), rotation='horizontal')
            plt.subplot(len(noises), nb_subs + len(nbs), 2 + j)
            plt.imshow(patterns[nb_samples], vmin=0, vmax=1+noises[-1])
            if j == 0:
                plt.title('subject 2')

        N = Ns
        # M = np.sqrt(M)
        M_image1 = M_image / (np.max(M_image) * N)
        bary_wass_image1, log = ot.bregman.barycenter(A, M_image1, reg, weights,
                                                      numItermax=1000, log=True)
        plt.subplot(len(noises), nb_subs + len(nbs), nb_samples+ 2 + j)
        plt.imshow(bary_wass_image1.reshape(patterns[0].shape))
        if j == 0:
            plt.title('barycenter\n{} images'.format(nb_samples*2))
    j += nb_subs + len(nbs)



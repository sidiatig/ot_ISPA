import numpy as np
import ot
import matplotlib.pyplot as plt

# two zones in one image, two images are similar
x_min = [10, 70]
x_max = [20, 80]
y_min = [5, 40]
y_max = [10, 45]

x_size = 100
y_size = 50

nb_zones = len(x_min)
nb_subs = 2
# nb of samples in each sub
nb_samples = 1

# Ns is the weight for distance matrix
Ns = [0.5]
noises = [0.001, 0.01, 0.05, 0.1, 0.5]
reg = 1 / 800
alpha = 1 / (nb_subs * nb_samples)  # 0<=alpha<=1
weights = np.ones(nb_subs * nb_samples) * alpha

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



gap1 = [0, 3]
gap2 = [0, -1] #-1,-5,-15,-30
# noisefrees only have the activated zone
noisefrees = np.zeros([nb_subs, x_size, y_size])
for sub_id in range(nb_subs):
    for zone_id in range(nb_zones):
        if zone_id == 0:
            noisefrees[sub_id][x_min[zone_id] + gap1[sub_id] :x_max[zone_id] + gap1[sub_id],
            y_min[zone_id]- gap1[sub_id]:y_max[zone_id] - gap1[sub_id]] = 1
        else:
            noisefrees[sub_id][x_min[zone_id]:x_max[zone_id],
            y_min[zone_id]+ gap2[sub_id]:y_max[zone_id]+ gap2[sub_id]] = 1
        # noisefrees[sub_id] += 0.00001
plt.figure()
plt.subplot(1,2,1)
plt.imshow(noisefrees[0])
plt.subplot(1,2,2)
plt.imshow(noisefrees[1])


plt.figure()
fig = plt.gcf()
fig.suptitle('two activated zones with gaussian noise', fontsize=14)
j = 0
# nb_samples for each sub
for noise in noises:
    patterns = np.zeros([nb_subs * nb_samples, x_size, y_size])
    for sub_id in range(nb_subs):
        for sample_id in range(nb_samples):
            patterns[sample_id + sub_id * nb_samples] = \
                noisefrees[sub_id] + noise * np.random.random_sample((x_size, y_size))

    arti = patterns.reshape([-1, x_size*y_size])
    arti_dis = np.empty(arti.shape)
    for i in range(arti.shape[0]):
        data = arti[i] - np.min(arti[i])
        data /= np.sum(data)
        arti_dis[i] = data
    A = np.transpose(arti_dis)
    # arti = np.transpose(arti)

    plt.subplot(len(noises), nb_subs + len(Ns), 1 + j)
    plt.imshow(patterns[0], vmin=0, vmax=1+noises[-1])
    if j == 0:
        plt.title('subject 1'.format(noise))
    plt.ylabel('noise:\n {}'.format(noise), rotation='horizontal')
    plt.subplot(len(noises), nb_subs + len(Ns), 2 + j)
    plt.imshow(patterns[nb_samples], vmin=0, vmax=1+noises[-1])
    if j == 0:
        plt.title('subject 2'.format(noise))
    for i in range(len(Ns)):
        N = Ns[i]
        # M = np.sqrt(M)
        M_image1 = M_image / (np.max(M_image) * N)
        bary_wass_image1, log = ot.bregman.barycenter(A, M_image1, reg, weights,
                                                      numItermax=1000, log=True)
        plt.subplot(len(noises), 2 + len(Ns), i + 3 + j)
        plt.imshow(bary_wass_image1.reshape(patterns[0].shape))
        if j == 0:
            plt.title('barycenter\n{}'.format(N))
    j += nb_subs + len(Ns)



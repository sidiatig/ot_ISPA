from os import system

code_dir = '/hpc/crise/wang.q/src_code/Pycharm_Projects/ot_ISPA/scripts'

noise_type = 'gaus'
# regs = [0.01, 0.003, 0.001, 0.0008, 0.0005]  # 0.001 is the best
# cs = [-4] # step for gradient
# bary_type = 'bregman'
# # n_samples = [i for i in range(1,6)] # n_samples is the nb of samples in each subject
# noises = [0.0001, 0.0005, 0.001, 0.005, 0.01]
# Ns = [0.5]
# n_samples = [i for i in range(1,20,3)] # add more sample each subject, min, max, snr don't change
regs = [0.001]
cs = [-4] # step for gradient
bary_type = 'bregman'
# n_samples = [i for i in range(1,6)] # n_samples is the nb of samples in each subject
noises = [0.006]
Ns = [0.5]
n_samples = [i for i in range(4,20,3)]

for reg in regs:
    for c in cs:
        for noise in noises:
            for N in Ns:
                for nb in n_samples:
                    cmd = "frioul_batch -n '11,12,13,14,15,16,17' -c 3 " \
                          "'/hpc/crise/anaconda3/bin/python3.5 " \
                          "%s/frioul_dual_barycenter.py %s %f %f %f %d %s %f'" \
                          % (code_dir, noise_type, reg, c, noise, nb, bary_type, N)
                    # a = commands.getoutput(cmd)
                    a = system(cmd)
                    print(cmd)

# a = np.load(result_dir + '/artificial_gaus_0.001noise_4samples_0.001reg_-4.0c_0.5Ns_bregman.npz')['barycenter']
# b = np.load(result_dir + '/artificial_gaus_0.001noise_4samples_0.001reg_-4.0c_0.5Ns_bregman.npz')['barycenter']
# a = a.reshape((100, 50))
# b = b.reshape((100, 50))
# a_rest = []
# b_rest = []
# for i in range(100):
#     for j in range(50):
#         if i in list(range(40, 50)) and j in list(range(23, 28)):
#             pass
#         else:
#             a_rest.append(a[i, j])
#             b_rest.append(b[i, j])
# a_snr = np.mean(a[40:50, 23:28].flatten()) / np.std(a_rest)
# b_snr = np.mean(b[40:50, 23:28].flatten()) / np.std(b_rest)
# print(a.max(), a.min(), a_snr)
# print(b.max(), b.min(), b_snr)

# for k in range(1, 20, 3):
#     a = np.load(result_dir + '/artificial_gaus_0.002noise_2subs_{}samples_1zones_0.001reg_-4.0c_0.5Ns_difloc_matrixdist_bregman.npz'.format(k))[
#         'barycenter']
#     a = a.reshape((100, 50))
#     a_rest = []
#     a_signal = []
#     for i in range(100):
#         for j in range(50):
#             if i in list(range(10, 20)) and j in list(range(5, 10)):
#                 a_signal.append(a[i,j])
#             else:
#                 a_rest.append(a[i, j])
#     a_snr = np.mean(a_signal) / np.std(a_rest)
#     print(k, '&', a.max(), '&', a.min(),'&', a_snr, '&', np.mean(a_signal), '&', np.std(a_rest), '\cr\hline')


# a = np.load(result_dir + '/artificial_gaus_0.0001noise_19samples_0.001reg_-4.0c_0.5Ns_bregman.npz')['patterns'][0]
# b = np.load(result_dir + '/artificial_gaus_0.0001noise_19samples_0.0008reg_-4.0c_0.5Ns_bregman.npz')['barycenter']
# a = a.reshape((100, 50))
# b = b.reshape((100, 50))
# a_rest = []
# for i in range(100):
#     for j in range(50):
#         if i in list(range(10, 20)) and j in list(range(5, 10)):
#             pass
#         else:
#             a_rest.append(a[i, j])
#
# b_rest = []
# for i in range(100):
#     for j in range(50):
#         if i in list(range(40, 50)) and j in list(range(23, 28)):
#             pass
#         else:
#             b_rest.append(b[i, j])
# a_snr = np.mean(a[10:20, 5:10].flatten()) / np.std(a_rest)
# b_snr = np.mean(b[40:50, 23:28].flatten()) / np.std(b_rest)
# print(a.max(), a.min(), a_snr)
# print(b.max(), b.min(), b_snr)

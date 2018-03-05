import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
#only use data from logistic regression

# result_dir = '/hpc/crise/wang.q/results/ISPA/'
# datasets = []
# clfmets = []
# exps = []
# datamets = []
# methods = []
#
# clf_met = 'logis'
# for exp,d,clf1,subn,clfn in zip(['fmril', 'fmrir', 'meg'],[20,20,170],
#                                 [90,90,14],[60,60,6],[60,60,50]):
#     split_nb = 120 if exp == 'meg' else 50
#     result = result_dir + 'result/'
#     for datamet in ['indi', 'no']:
#         for method in ['subspace','single','multi','baseline']:
#             if method == 'subspace':
#
#                 data = np.load(result+'subspace_{}_{}_{}_{}dimens_{}splits_gridsearch.npy'.
#                                format(exp,datamet,clf_met,d,split_nb))
#             elif method == 'single':
#                 data = np.load(result+'decision_{}_{}_{}_1subs_{}clf_{}splits_gridsearch.npy'.
#                                format(exp,datamet,clf_met,clf1,split_nb))
#             elif method == 'multi':
#                 data = np.load(result + 'decision_{}_{}_{}_{}subs_{}clf_{}splits_gridsearch.npy'.
#                                format(exp, datamet, clf_met, subn, clfn,split_nb))
#             else:
#                 data = np.load(result + 'nor_{}_{}_{}_{}splits_gridsearch.npy'.
#                                format(exp,datamet,clf_met,split_nb))
#
#             datasets.append(data)
#             clfmets += [clf_met] * split_nb
#             exps += [exp] * split_nb
#             datamets += [datamet] * split_nb
#             methods += [method] * split_nb
#
# datasets = np.hstack(datasets)
# print(len(datasets),len(clfmets),len(exps),len(datamets),len(methods))
#
# datatitle = {'classifier':[],'experiment':[],'datamethod':[],'accuracy':[],'method':[]}
# dataframe = pd.DataFrame(datatitle)
# dataframe.classifier = clfmets
# dataframe.experiment = exps
# dataframe.datamethod = datamets
# dataframe.accuracy = datasets
# dataframe.method = methods
# dataframe.to_csv(result_dir+'plot_experiment2.csv',index=False)
#

# def stars(p):
#     if p < 0.0001:
#         return "****"
#     elif (p < 0.001):
#         return "***"
#     elif (p < 0.01):
#         return "**"
#     elif (p < 0.05):
#         return "*"
#     else:
#         return "-"

result_dir = '/hpc/crise/wang.q/results/ISPA/'
results = pd.read_csv(result_dir + 'plot_experiment3.csv')
results = results[results.datamethod=='no']
results = results[(results['method']=='subspace') | (results['method']=='baseline')]



# plt.close('all')
# plt.rcParams['ytick.major.pad'] = 2
# plt.rcParams['ytick.labelsize'] = 12.
# sns.set_style("whitegrid",
#                   {"xtick.color": '0',
#                    "ytick.color": '0',
#                    "text.color": '0',
#                    "grid.color": '.98',
#                    "axes.edgecolor": '.7'})
# plt.figure(figsize=(9,5))
# fontsize = 12.5
# fontsize2 = 11
# text_fontsize = 11



datasets = []
for experiment in ['fmril', 'fmrir', 'meg']:
    data = results.loc[results['experiment']==experiment]
    data = [np.array(data[data['method']=='subspace'].accuracy),
         np.array(data[data['method']=='baseline'].accuracy)]
    datasets.append(data)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 3))
box = axes[0].boxplot(datasets[0], vert=False, widths=0.3)
axes[0].set_title('fMRI_1')
box['fliers'].set_markersize(40)


axes[1].boxplot(datasets[1], vert=False, widths=0.3)
axes[1].set_title('fMRI_2')


axes[2].boxplot(datasets[2], vert=False, widths=0.3)
axes[2].set_title('MEG')

for ax in axes:
    ax.xaxis.grid(color='b', linestyle='-', linewidth=0.2)
    ax.set_xlabel('accuracy')

plt.setp(axes[0], yticks=[y+1 for y in range(len(datasets[0]))],
         yticklabels=['subspace\nalignment', 'no\nnormalization'])
for i in range(1,3):
    plt.setp(axes[i], yticks=[y+1 for y in range(len(datasets[0]))],
             yticklabels=['', ''])
plt.tight_layout()


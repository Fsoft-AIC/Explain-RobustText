import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import matplotlib.pylab as pylab

parser = ArgumentParser()
parser.add_argument('--compared_metric', type=str, choices=['rmse','r2','mae','evs','mape'], 
                                          help='Choose a metric to be compared', default='rmse')
parser.add_argument('--compared_model', type=str, choices=['bert','distil_roberta'], 
                                          help='Choose a model to be compared', default='bert')

args = parser.parse_args()
pylab.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8,3),dpi=280)

# Creating dataset
data_extra_bert = np.load(f'{args.compared_metric}_extra_bert.npy')
data_inter_bert = np.load(f'{args.compared_metric}_inter_bert.npy')
data_rf_bert = np.load(f'{args.compared_metric}_rf_verify_bert.npy')

data_bert = [data_extra_bert, data_inter_bert, data_rf_bert]
ax_bert = fig.add_subplot(121)
 
# Creating axes instance
bp_bert = ax_bert.boxplot(data_bert, patch_artist = True,
                notch ='True', 
                vert = 0
                )
 
colors = ['#0000FF', '#00FF00', '#FFFF00']
 
for patch, color in zip(bp_bert['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp_bert['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp_bert['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp_bert['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp_bert['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax_bert.set_yticklabels(['Extrapolation','Interpolation','Adversarial\nTraning'])
 
# Adding title
metric = {'rmse':'RMSE','r2':'R2','mae':'MAE','evs':'EVS','mape':'MAPE'}
plt.title(f"{metric[args.compared_metric]} BERT")
 
# Removing top axes and right axes
# ticks
ax_bert.get_xaxis().tick_bottom()
ax_bert.get_yaxis().tick_right()

# Creating dataset
data_extra_roberta = np.load(f'{args.compared_metric}_extra_distil_roberta.npy')
data_inter_roberta = np.load(f'{args.compared_metric}_inter_distil_roberta.npy')
data_rf_roberta = np.load(f'{args.compared_metric}_rf_verify_distil_roberta.npy')

data_roberta = [data_extra_roberta, data_inter_roberta, data_rf_roberta]
ax_roberta = fig.add_subplot(122)
 
# Creating axes instance
bp_roberta = ax_roberta.boxplot(data_roberta, patch_artist = True,
                notch ='True', 
                vert = 0
                )
 
colors = ['#0000FF', '#00FF00', '#FFFF00']
 
for patch, color in zip(bp_roberta['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp_roberta['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp_roberta['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp_roberta['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp_roberta['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax_roberta.set_yticklabels([])

# Adding title
metric = {'rmse':'RMSE','r2':'R2','mae':'MAE','evs':'EVS','mape':'MAPE'}
plt.title(f"{metric[args.compared_metric]} RoBERTa")
 
# Removing top axes and right axes
# ticks
ax_roberta.get_xaxis().tick_bottom()

# save and show plot
plt.tight_layout()
plt.savefig(f'image/compare_experiments/compare_{args.compared_metric}_unify.png')
plt.show()
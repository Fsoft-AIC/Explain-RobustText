import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
from argparse import ArgumentParser
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,explained_variance_score,\
                            mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import itertools
from sklearn.inspection import permutation_importance
from utils.PyALE import ale
import statsmodels.api as sm
import random
from statistics import mean 
import time
import seaborn as sns

# parser = ArgumentParser()
# parser.add_argument('--sample_data_file', type=str, choices=['data_roberta.csv','data_bert.csv'], 
#                                           help='Sampled data file', default='data_bert.csv')
# parser.add_argument('--perturb_sample_data_file', type=str, choices=['data_roberta - Copy.csv','data_bert - Copy.csv'], 
#                                           help='Sampled data file', default='data_bert - Copy.csv')
# parser.add_argument('--r2_threshold_inter', type=float, default=0.3, help='Folder in which to save model and logs')
# parser.add_argument('--min_iters_inter', type=int, default=200, help='Folder in which to save model and logs')
# args = parser.parse_args()

datasets = ['amazon_review_full', # 18
            'amazon_review_polarity','dbpedia', # 56, 71
            'yahoo_answers','ag_news', # 11, 49
            'yelp_review_full','yelp_review_polarity', # 2, 69
            'banking77__2', 'banking77__4', 'banking77__5', # 2, 5, 3
            'banking77__10', 'banking77__14', # 4, 3
            'tweet_eval_emoji_2', 'tweet_eval_emoji_4', 'tweet_eval_emoji_5', # 4, 1, 3
            'tweet_eval_emoji_10', 'tweet_eval_emoji_14' # 5, 2
           ]
attackers = ['ASR_TextFooler','ASR_PWWS','ASR_BERT','ASR_DeepWordBug']

def sample_ASR(data_file, n_sample):
    file = pd.read_csv(data_file,sep=',')

    file = file[file.notnull().all(1)].drop(columns='ASR_BERT')
    file = file[(file!='Nan').all(1)]

    file['Fisher ratio'] = file['Fisher ratio'].apply(lambda x:1/x)
    file.rename(columns = {'Fisher ratio':'FR', 'CalHara Index':'CHI',
                           'DaBou Index':'DBI', 'Pearson Med':'PMS',
                           'Mean distance':'MD', 'Minimum number of tokens': 'Min # tokens',
                           'Maximum number of tokens': 'Max # tokens', 'Number of cluster': '# clusters', 'Kurtosis': 'KTS',
                           'Average number of tokens': 'Avg. # tokens', 'Number of unique tokens': '# unique tokens',
                           'Misclassification rate': 'MR', 'Number of classes': '# classes',
                           'Number of labels': '# labels',}, inplace = True)
    file = file.astype({'ASR_DeepWordBug': 'float64','ASR_PWWS': 'float64','ASR_TextFooler': 'float64'})
    file['ASR']=(file['ASR_TextFooler']+file['ASR_PWWS']+file['ASR_DeepWordBug'])/3
    file.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_DeepWordBug'],inplace=True)

    def convert_dataset(x):
        if x[:5]=='banki':
            return 'banking77'
        elif x[:5]=='tweet':
            return 'tweet_eval_emoji'
        return x
    file['Dataset'] = file['Dataset'].map(convert_dataset)
    file = file.sort_values(by=['Index']).reset_index(drop=True)

    supp_file = pd.read_csv('supplementary_2x.csv',sep=',',index_col=False)
    supp_file = supp_file[supp_file.notnull().all(1)]
    supp_file = supp_file[(supp_file!='Nan').all(1)]
    supp_file.sort_values(by=['Index'])
    supp_file.rename(columns = {'Fisher ratio':'FR', 'CalHara Index':'CHI',
                           'DaBou Index':'DBI', 'Pearson Med':'PMS',
                           'Mean distance':'MD', 'Minimum number of tokens': 'Min # tokens',
                           'Maximum number of tokens': 'Max # tokens', 'Number of cluster': '# clusters', 'Kurtosis': 'KTS',
                           'Average number of tokens': 'Avg. # tokens', 'Number of unique tokens': '# unique tokens',
                           'Misclassification rate': 'MR', 'Number of classes': '# classes',
                           'Number of labels': '# labels',}, inplace = True)
    perturb_file = file.copy(deep=True).reset_index(drop=True)

    temp = supp_file[supp_file['Index'].isin(perturb_file['Index'].values)].reset_index(drop=True)
    temp = temp[['MD','FR','CHI',
                 'DBI','# clusters']]
    perturb_file[['MD','FR','CHI',
                 'DBI','# clusters']] = temp

    print('*-'*100)
    print('Embedding Verification')

    datasets = [
                'amazon_review_full', # 18
                'amazon_review_polarity','dbpedia', # 56, 71
                'yahoo_answers','ag_news', # 11, 49
                'yelp_review_full','yelp_review_polarity', # 2, 69
                'banking77', 'tweet_eval_emoji'
               ]


    ############### Extrapolation Experiment ###############
    pylab.rcParams['font.size'] = 15

    print('-*'*100)
    print('Extrapolation Experiment')

    pred_y = []
    pred_y_adv = []
    for t in range(n_sample):
        test_dataset = random.sample(datasets,1)
        ind_train = file[~file['Dataset'].isin(test_dataset)].index.tolist()
        ind_test = file[file['Dataset'].isin(test_dataset)].index.tolist()
        data_train, data_test = file.iloc[ind_train], file.iloc[ind_test]
        x_train, y_train = data_train.drop(columns=['Dataset','ASR']), np.array(data_train['ASR'])
        x_test, x_test_adv = data_test.drop(columns=['Dataset','ASR']), perturb_file.iloc[ind_test].drop(columns=['Dataset','ASR'])

        # Random Forest
        rdfr_rgs = RandomForestRegressor(max_depth=20, random_state=0).fit(x_train,y_train)
        start = time.time()
        predicted_y = rdfr_rgs.predict(x_test)
        predicted_y_adv = rdfr_rgs.predict(x_test_adv)
        pred_y.append(np.mean(predicted_y))
        pred_y_adv.append(np.mean(predicted_y_adv))
    return pred_y, pred_y_adv

pred_y_bert, pred_y_adv_bert = sample_ASR('data_bert.csv',100)
pred_y_robert, pred_y_adv_robert = sample_ASR('data_roberta.csv',100)
data = [pred_y_bert, pred_y_adv_bert, pred_y_robert, pred_y_adv_robert]

fig = plt.figure(figsize=(8,2),dpi=280)
ax = fig.add_subplot(111)

 
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch ='True', 
                vert = 0
                )
 
colors = ['#0000FF', '#00FF00', '#FFFF00']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_xlabel('Predicted ASR')
ax.set_yticklabels(['Original (BERT)','Perturb (BERT)','Original (RoBERTa)', 'Perturb (RoBERTa)'])
 
# Adding title
plt.title(f"Extrapolation")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

fig.tight_layout()
fig.savefig(f'image/embedding_verify/embedding_verify_extra_2x.png')
fig.show()
plt.show()
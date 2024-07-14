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
    file.drop(columns=['Dataset'],inplace=True)
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


    ############### Interpolation Experiment ###############
    pylab.rcParams['font.size'] = 15

    print('-*'*100)
    print('Interpolation Experiment')


    ale_func_inter = None
    base_r2 = -1000
    ale_inter_x_test, ale_inter_y_test = None, None
    pred_y = []
    pred_y_adv = []

    for t in range(n_sample):
        ind_train, ind_test = train_test_split(range(len(file.index)),test_size = 0.3, random_state = int(time.time()))
        data_train, data_test = file.iloc[ind_train], file.iloc[ind_test]
        x_train, y_train = data_train.drop(columns='ASR'), np.array(data_train['ASR'])
        x_test, x_test_adv = data_test.drop(columns='ASR'), perturb_file.iloc[ind_test].drop(columns='ASR')

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
fig, ax = plt.subplots(figsize=(4,3), dpi=200)
sns.histplot(data=pred_y_bert, label='Original (BERT)', binwidth=0.02, stat = 'probability', kde = True, palette = "Spectral", color = 'red', alpha = 0.5, ax=ax)
sns.histplot(data=pred_y_adv_bert, label='Perturb (BERT)', binwidth=0.02, stat = 'probability', kde = True, palette = "Spectral", color = 'green', alpha = 0.5, ax=ax)
sns.histplot(data=pred_y_robert, label='Original (RoBERTa)', binwidth=0.02, stat = 'probability', kde = True, palette = "Spectral", color = 'blue', alpha = 0.5, ax=ax)
sns.histplot(data=pred_y_adv_robert, label='Perturb (RoBERTa)', binwidth=0.02, stat = 'probability', kde = True, palette = "Spectral", color = 'orange', alpha = 0.5, ax=ax)
ax.set_xlabel('Predicted ASR')
ax.set_title('Interpolation')
fig.legend(fontsize='small', bbox_to_anchor=(0.85, 0.87))
fig.tight_layout()
fig.savefig(f'image/embedding_verify/embedding_verify_inter_2x.png')
fig.show()
plt.show()
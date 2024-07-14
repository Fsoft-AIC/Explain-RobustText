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
                            mean_absolute_percentage_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
import itertools
from sklearn.inspection import permutation_importance
from utils.PyALE import ale
import statsmodels.api as sm
import random
from statistics import mean 
import time

'''
BERT
[ 7  2 11 12  5 13 10  9  0  4  8  6  3  1]
[ 7  2 11  8  9  0  3  5 12  4  6 10 13  1]
RoBERTa
[11  8  7  9 12 13 10  2  5  4  0  3  6  1]
[ 6  3 12  1  0  9  7 13 10  4  2  5 11  8]
ELECTRA
[11 12  7  8  5  4 10 13  0  2  9  3  6  1]
GPT2
[11  2  8  7  9 13 10  4  5  0 12  3  6  1]
[ 2 12  8  3  9 11  7 10 13  5  4  0  6  1]
BART
[11  7  2 12  5 10 13  4  0  9  3  8  6  1]
[ 6  7 11  3  4 13 10  1  8  5  9  2 12  0]
-----------
BERT
[ 9  2  1  3 13 10  5 11  6 12  7  0  4  8]
RoBERTa
[ 6  3 12  1  0  9  7 13 10  4  2  5 11  8]
ELECTRA
[12  6  2 11  3  9 10 13  0  7  1  4  8  5]
GPT2
[ 2 12  8  3  9 11  7 10 13  5  4  0  6  1]
BART
[ 6  7 11  3  4 13 10  1  8  5  9  2 12  0]
-----------
Most: 2 3 
Features:
['Avg. # tokens', '# unique tokens', 'Min # tokens', 'Max # tokens',
 'MD', 'FR', 'CHI', 'DBI', '# clusters', 'PMS', '# labels', 'KTS', 'MR', '# classes']
'''

parser = ArgumentParser()
parser.add_argument('--sample_data_file', type=str, choices=['data_roberta.csv','data_bert.csv','data_electra.csv','data_gpt2.csv','data_bart.csv'], 
                                          help='Sampled data file', default='data_gpt2.csv')
parser.add_argument('--r2_threshold_inter', type=float, default=0.3, help='Folder in which to save model and logs')
parser.add_argument('--min_iters_inter', type=int, default=200, help='Folder in which to save model and logs')
args = parser.parse_args()

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


# args.sample_data_file: ['data_roberta.csv','data_bert.csv']
file = pd.read_csv(args.sample_data_file,sep=',')
file = file[file.notnull().all(1)].drop(columns='ASR_BERT')
file = file[(file!='Nan').all(1)]

new_file = file[file['Dataset']=='amazon_review_full']

file.drop(columns=['Index'],inplace=True)
file = file.astype({'ASR_DeepWordBug': 'float64','ASR_PWWS': 'float64','ASR_TextFooler': 'float64'})
file['ASR']=(file['ASR_TextFooler']+file['ASR_PWWS']+file['ASR_DeepWordBug'])/3
file = file[(file['ASR_DeepWordBug'] >= 0.1) & (file['ASR_PWWS'] >= 0.1) & (file['ASR_TextFooler'] >= 0.1)]
file['Fisher ratio'] = file['Fisher ratio'].apply(lambda x:1/x)
file.rename(columns = {'Fisher ratio':'FR', 'CalHara Index':'CHI',
                       'DaBou Index':'DBI', 'Pearson Med':'PMS',
                       'Mean distance':'MD', 'Minimum number of tokens': 'Min # tokens',
                       'Maximum number of tokens': 'Max # tokens', 'Number of cluster': '# clusters', 'Kurtosis': 'KTS',
                       'Average number of tokens': 'Avg. # tokens', 'Number of unique tokens': '# unique tokens',
                       'Misclassification rate': 'MR', 'Number of classes': '# classes',
                       'Number of labels': '# labels',}, inplace = True)

file.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_DeepWordBug','# labels'],inplace=True)
print(file.columns) # (# labels == # classes)

def convert_dataset(x):
    if x[:5]=='banki':
        return 'banking77'
    elif x[:5]=='tweet':
        return 'tweet_eval_emoji'
    return x

file['Dataset'] = file['Dataset'].map(convert_dataset)
datasets = [
            'amazon_review_full', # 18
            'amazon_review_polarity','dbpedia', # 56, 71
            'yahoo_answers','ag_news', # 11, 49
            'yelp_review_full','yelp_review_polarity', # 2, 69
            'banking77', 'tweet_eval_emoji'
           ]


############### Interpolation Experiment ###############
pylab.rcParams['font.size'] = 17

print('-*'*100)
print('Interpolation Experiment')
file.drop(columns=['Dataset'],inplace=True)

r2 = 0
r2_base = -1000
x_test_bst = None
summary_bst = None
rdfr_bst = None
for t in itertools.count():
    file = file.sample(frac=1)
    x_train, x_test, y_train, y_test = train_test_split(file.drop(columns='ASR'), np.array(file['ASR']), test_size = 0.6, random_state = 0)
    x_train_skl = x_train
    y_train_skl = y_train

    # Random Forest
    rdfr_rgs = RandomForestRegressor(max_depth=20, random_state=0).fit(x_train_skl,y_train_skl)
    summary = x_test.copy()
    summary['Predicted'] = rdfr_rgs.predict(x_test)
    summary['Groundtruth'] = y_test
    summary['error'] = abs(summary['Groundtruth'] - summary['Predicted'])
    summary = summary.sort_values(by='error',ascending=True).reset_index(drop=True)
    '''
    columns
    ['Avg. # tokens', '# unique tokens', 'Min # tokens', 'Max # tokens',
       'MD', 'FR', 'CHI', 'DBI', '# clusters', 'PMS', '# labels', 'KTS', 'MR',
       '# classes', 'Predicted', 'Groundtruth', 'error']
    '''
    r2 = r2_score(y_test, summary['Predicted'])

    if r2 > r2_base:
        r2_base = r2
        x_test_bst = x_test.copy()
        summary_bst = summary.copy()
        rdfr_bst = rdfr_rgs
    if t == 200:
        break
    # if file.groupby('# classes')['ASR'].max()

pred = rdfr_bst.predict(file.drop(columns='ASR'))
temp = pd.DataFrame({'Classes': file['# classes'], 'Error': abs(pred-file['ASR'])})
print(temp.groupby('Classes')['Error'].mean())
# Plot the points
plt.scatter(file['# classes'], abs(pred-file['ASR']), color='blue', marker='o', label='Points')
# Show the plot
plt.show()

X = x_test_bst
y = summary_bst['error']
quantile_error = y.quantile(0.7)
y = (y > quantile_error).astype(bool)
X_norm = preprocessing.MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X_norm, columns=X.columns)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression(max_iter=100)
log_reg_model.fit(X_train, y_train)  

y_pred_train = log_reg_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", train_accuracy)

y_pred_test = log_reg_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Testing Accuracy:", test_accuracy)

# print(X.columns)
# print(np.argsort(abs(log_reg_model.coef_[0]))[::-1])
# print(abs(log_reg_model.coef_))
print(file.groupby('# classes')['ASR'].mean())
print(file.groupby('# classes')['ASR'].var())


print('Finish')
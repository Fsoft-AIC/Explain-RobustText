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

parser = ArgumentParser()
parser.add_argument('--sample_data_file', type=str, choices=['data_roberta.csv','data_bert.csv','data_bart.csv','data_electra.csv','data_gpt2.csv'], 
                                          help='Sampled data file', default='data_bert.csv')
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
if args.sample_data_file == 'data_bert.csv':
    model = '_bert'
elif args.sample_data_file == 'data_roberta.csv':
    model = '_distil_roberta'
elif args.sample_data_file == 'data_electra.csv':
    model = '_electra'
elif args.sample_data_file == 'data_gpt2.csv':
    model = '_gpt2'
elif args.sample_data_file == 'data_bart.csv':
    model = '_bart'
else:
    raise Exception('Wrong data file.')

file = file[file.notnull().all(1)].drop(columns='ASR_BERT')
file = file[(file!='Nan').all(1)]

new_file = file[file['Dataset']=='amazon_review_full']

file.drop(columns=['Index'],inplace=True)
file = file.astype({'ASR_DeepWordBug': 'float64','ASR_PWWS': 'float64','ASR_TextFooler': 'float64'})
file['ASR']=(file['ASR_TextFooler']+file['ASR_PWWS']+file['ASR_DeepWordBug'])/3
file = file[(file['ASR_DeepWordBug'] >= 0.1) & (file['ASR_PWWS'] >= 0.1) & (file['ASR_TextFooler'] >= 0.1)]
file['Fisher ratio'] = file['Fisher ratio'].apply(lambda x:1/x)
file.drop(columns=['Number of labels'],inplace=True)
file.rename(columns = {'Fisher ratio':'FR', 'CalHara Index':'CHI',
                       'DaBou Index':'DBI', 'Pearson Med':'PMS',
                       'Mean distance':'MD', 'Minimum number of tokens': 'Min # tokens',
                       'Maximum number of tokens': 'Max # tokens', 'Number of cluster': '# clusters', 'Kurtosis': 'KTS',
                       'Average number of tokens': 'Avg. # tokens', 'Number of unique tokens': '# unique tokens',
                       'Misclassification rate': 'MR', 'Number of classes': '# classes',
                       'Number of labels': '# labels',}, inplace = True)

inspect_ft = '# unique tokens'

# Plot the points
plt.scatter(file[inspect_ft], file['ASR'], color='blue', marker='o', label='Points')
# Show the plot
plt.show()

# file = file[((file[inspect_ft] < 2000) & (file['ASR'] > 0.8)) |
#             ((file[inspect_ft] > 2000) & (file[inspect_ft] < 5000) & (file['ASR'] > 0.6)) |
#             ((file[inspect_ft] > 6000) & (file[inspect_ft] < 8000) & (file['ASR'] > 0.4) & (file['ASR'] < 0.5)) |
#             ((file[inspect_ft] > 8000) & (file[inspect_ft] < 10000) & (file['ASR'] < 0.4))
#             ]

file.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_DeepWordBug'],inplace=True)

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


ale_func_inter = None
base_r2 = -1000
ale_inter_x_test, ale_inter_y_test = None, None
rmse_gb,rmse_mlp,rmse_lr,rmse_rf = [],[],[],[]
r2_gb,r2_mlp,r2_lr,r2_rf = [],[],[],[]
mae_gb,mae_mlp,mae_lr,mae_rf = [],[],[],[]
evs_gb,evs_mlp,evs_lr,evs_rf = [],[],[],[]
mape_gb,mape_mlp,mape_lr,mape_rf = [],[],[],[]

if args.sample_data_file == 'data_bert.csv':
    title = 'BERT'
elif args.sample_data_file == 'data_roberta.csv':
    title = 'RoBERTa'
elif args.sample_data_file == 'data_electra.csv':
    title = 'Electra'
elif args.sample_data_file == 'data_gpt2.csv':
    title = 'GPT2'
elif args.sample_data_file == 'data_bart.csv':
    title = 'BART'
else:
    raise Exception('Wrong data file.')
title = title + ' Interpolation'

for t in itertools.count():
    file = file.sample(frac=1)
    x_train, x_val, y_train, y_val = train_test_split(file.drop(columns='ASR'), np.array(file['ASR']), test_size = 0.4, random_state = 0)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.2, random_state = 0)
    
    # Gradient Boosting
    gb_rgs = lgb.LGBMRegressor(learning_rate=0.05, max_bin=400,
                              metric='rmse', n_estimators=5000,
                              objective='regression', random_state=79,
                              )

    gb_rgs.fit(x_train, y_train,
        eval_set = [(x_val, y_val)],
        eval_metric = ['rmse'],
        callbacks=[lgb.early_stopping(10)])
    # gb_rgs.fit(file.drop(columns='ASR'), np.array(file['ASR']),
    #     eval_set = [(x_val, y_val)],
    #     eval_metric = ['rmse'],
    #     callbacks=[lgb.early_stopping(10)])
    predicted_y = gb_rgs.predict(x_test)
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(predicted_y.shape[0]):
        summary['Predicted'].append(predicted_y[i])
        summary['Groundtruth'].append(y_test[i])
    summary = pd.DataFrame(summary)
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*50)
    rmse_gb.append(mean_squared_error(y_test, predicted_y,squared=False))
    r2_gb.append(r2_score(y_test, predicted_y))
    mae_gb.append(mean_absolute_error(y_test, predicted_y))
    evs_gb.append(explained_variance_score(y_test, predicted_y))
    mape_gb.append(mean_absolute_percentage_error(y_test, predicted_y))
    
    # MLP
    x_train_skl = pd.concat([x_train,x_val])
    y_train_skl = np.concatenate((y_train,y_val))
    mlp_rgs = MLPRegressor(hidden_layer_sizes=(100,100), random_state=10, max_iter=5000).fit(x_train_skl, y_train_skl)
    predicted_y = mlp_rgs.predict(x_test)
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(predicted_y.shape[0]):
        summary['Predicted'].append(predicted_y[i])
        summary['Groundtruth'].append(y_test[i])
    summary = pd.DataFrame(summary)
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*50)
    rmse_mlp.append(mean_squared_error(y_test, predicted_y,squared=False))
    r2_mlp.append(r2_score(y_test, predicted_y))
    mae_mlp.append(mean_absolute_error(y_test, predicted_y))
    evs_mlp.append(explained_variance_score(y_test, predicted_y))
    mape_mlp.append(mean_absolute_percentage_error(y_test, predicted_y))

    # Linear Regression
    ln_rgs = LinearRegression(fit_intercept=True).fit(x_train_skl,y_train_skl)
    predicted_y = ln_rgs.predict(x_test)
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(predicted_y.shape[0]):
        summary['Predicted'].append(predicted_y[i])
        summary['Groundtruth'].append(y_test[i])
    summary = pd.DataFrame(summary)
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*50)
    rmse_lr.append(mean_squared_error(y_test, predicted_y,squared=False))
    r2_lr.append(r2_score(y_test, predicted_y))
    mae_lr.append(mean_absolute_error(y_test, predicted_y))
    evs_lr.append(explained_variance_score(y_test, predicted_y))
    mape_lr.append(mean_absolute_percentage_error(y_test, predicted_y))

    # Random Forest
    rdfr_rgs = RandomForestRegressor(max_depth=20, random_state=0).fit(x_train_skl,y_train_skl)
    start = time.time()
    predicted_y = rdfr_rgs.predict(x_test)
    end = time.time()
    r2_rdfr = r2_score(y_test, predicted_y)
    if r2_rdfr > base_r2:
        ale_func_inter = rdfr_rgs
        base_r2 = r2_rdfr
        ale_inter_x_test, ale_inter_y_test = file.drop(columns='ASR'), np.array(file['ASR'])
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(predicted_y.shape[0]):
        summary['Predicted'].append(predicted_y[i])
        summary['Groundtruth'].append(y_test[i])
    summary = pd.DataFrame(summary)
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*50)
    rmse_rf.append(mean_squared_error(y_test, predicted_y,squared=False))
    r2_rf.append(r2_score(y_test, predicted_y))
    mae_rf.append(mean_absolute_error(y_test, predicted_y))
    evs_rf.append(explained_variance_score(y_test, predicted_y))
    mape_rf.append(mean_absolute_percentage_error(y_test, predicted_y))

    # Export results to CSV file
    if (max(r2_rf) > args.r2_threshold_inter or t > args.min_iters_inter):
        print('Feature Importance')
        print('*'*10)
        
        # Gradient Boosting FI
        print('Gradient Boosting FI')
        r = permutation_importance(ale_func_inter, ale_inter_x_test, ale_inter_y_test,
                                    n_repeats=100,
                                    random_state=0)

        important_ind = []
        for i in r.importances_mean.argsort()[::-1]:
            if len(important_ind) > 5:
                break
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                important_ind.append(i)
                print(f"{ale_inter_x_test.columns[i]:<8}: "
                      f"{r.importances_mean[i]:.3f}"
                     f" +/- {r.importances_std[i]:.3f}")
        fig, ax = plt.subplots(dpi=250)
        hbars = ax.barh(np.arange(len(important_ind)), 
                                    r.importances_mean[important_ind], 
                                    xerr=r.importances_std[important_ind],
                                    color='green',
                                    align='center')
        ax.set_yticks(np.arange(len(important_ind)), 
                                    labels=ale_inter_x_test.columns[important_ind])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Mean accuracy decrease")
        # ax.bar_label(hbars, fmt='%.2f')
        fig.tight_layout()
        fig.savefig(f'image/interpret/permutation/random_forest_permute_inter{model}.png')
        fig.show()
        plt.show()
        print('*'*10)
        break

global_report = pd.DataFrame([[mean(rmse_gb),max(rmse_gb),min(rmse_gb),np.var(rmse_gb),
                               mean(r2_gb),max(r2_gb),min(r2_gb),np.var(r2_gb),
                               mean(mae_gb),max(mae_gb),min(mae_gb),np.var(mae_gb),
                               mean(evs_gb),max(evs_gb),min(evs_gb),np.var(evs_gb),
                               mean(mape_gb),max(mape_gb),min(mape_gb),np.var(mape_gb)], 
                              [mean(rmse_lr),max(rmse_lr),min(rmse_lr),np.var(rmse_lr),
                               mean(r2_lr),max(r2_lr),min(r2_lr),np.var(r2_lr),
                               mean(mae_lr),max(mae_lr),min(mae_lr),np.var(mae_lr),
                               mean(evs_lr),max(evs_lr),min(evs_lr),np.var(evs_lr),
                               mean(mape_lr),max(mape_lr),min(mape_lr),np.var(mape_lr)],
                              [mean(rmse_mlp),max(rmse_mlp),min(rmse_mlp),np.var(rmse_mlp),
                               mean(r2_mlp),max(r2_mlp),min(r2_mlp),np.var(r2_mlp),
                               mean(mae_mlp),max(mae_mlp),min(mae_mlp),np.var(mae_mlp),
                               mean(evs_mlp),max(evs_mlp),min(evs_mlp),np.var(evs_mlp),
                               mean(mape_mlp),max(mape_mlp),min(mape_mlp),np.var(mape_mlp)],
                              [mean(rmse_rf),max(rmse_rf),min(rmse_rf),np.var(rmse_rf),
                               mean(r2_rf),max(r2_rf),min(r2_rf),np.var(r2_rf),
                               mean(mae_rf),max(mae_rf),min(mae_rf),np.var(mae_rf),
                               mean(evs_rf),max(evs_rf),min(evs_rf),np.var(evs_rf),
                               mean(mape_rf),max(mape_rf),min(mape_rf),np.var(mape_rf)]], 
                                columns=[   'RMSE_MEAN','RMSE_MAX','RMSE_MIN','RMSE_VAR',
                                            'R2_MEAN','R2_MAX','R2_MIN','R2_VAR',
                                            'MAE_MEAN','MAE_MAX','MAE_MIN','MAE_VAR',
                                            'EVS_MEAN','EVS_MAX','EVS_MIN','EVS_VAR',
                                            'MAPE_MEAN','MAPE_MAX','MAPE_MIN','MAPE_VAR'], 
                                index=['Gradient Boosting', 'Linear Regression', 'MLP', 'Random Forest'])

with open(f'rmse_inter{model}.npy', 'wb') as f:
    np.save(f, rmse_rf)
with open(f'r2_inter{model}.npy', 'wb') as f:
    np.save(f, r2_rf)
with open(f'mae_inter{model}.npy', 'wb') as f:
    np.save(f, mae_rf)
with open(f'evs_inter{model}.npy', 'wb') as f:
    np.save(f, evs_rf)
with open(f'mape_inter{model}.npy', 'wb') as f:
    np.save(f, mape_rf)

print(global_report)
(global_report.T).to_csv(f'result_summary_interpolate{model}.csv')


########## Accumulated Local Effects (ALE) for Interpolation ##########
discrete_fts = ['# unique tokens',
                'Min # tokens', 'Max # tokens', '# clusters',
                '# classes']
continuous_fts = ['Avg. # tokens', 'MD', 'FR', 
                  'CHI', 'DBI', 'PMS', 'KTS', 
                  'MR']

pylab.rcParams['font.size'] = 15
for i, ft in enumerate(discrete_fts+continuous_fts):
    fig = plt.figure(figsize=(6,2),dpi=250)
    axis = fig.add_subplot()
    ale_eff = ale(
        X=ale_inter_x_test, model=ale_func_inter, feature=[ft], grid_size=50, 
        feature_type='discrete' if ft in discrete_fts else 'continuous',
        include_CI=False, fig=fig,
        ax=axis
    ) # Keys: ['eff', 'size']
    
    plt.close()

    # Create a new figure and a single subplot
    fig, ax = plt.subplots(figsize=(5,3),dpi=250)

    x_values = ale_eff.index.tolist()
    y_values = ale_eff['eff'].tolist()

    # Create a line plot for discrete values
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='blue', label='Effect')

    # # Fit a polynomial regression line (1st degree) using numpy.polyfit
    # coefficients = np.polyfit(x_values, y_values, 1)
    # trendline = np.polyval(coefficients, x_values)

    # # Plot the trendline
    # plt.plot(x_values, trendline, linestyle='--', color='orange', label='Trend')

    if ft in discrete_fts:
        qt = int(max(x_values)-min(x_values))//3
        if qt > 0:
            x_ticks = [x_values[0],x_values[0]+qt,x_values[0]+qt*2,x_values[-1]]
        else:
            x_ticks = [x_values[0],x_values[-1]]
    else:
        if (max(x_values)-min(x_values)) > 1000:
            jump = (max(x_values)-min(x_values))//2
            x_ticks = [x_values[0],x_values[0]+jump,x_values[-1]]
        else:
            jump = (max(x_values)-min(x_values))//3
            x_ticks = [x_values[0],x_values[0]+jump,x_values[0]+jump*2,x_values[-1]]
    
    ax.set_xticks(x_ticks) # set new tick positions
    ax.margins(x=0) # set tight margins

    # Adding labels, title, and legend
    ax.set_xlabel(ft)
    ax.set_ylabel('Effect on prediction')
    ax.set_title(f"{title}")
    # fig.legend(fontsize='small', bbox_to_anchor=(0.85, 0.8))
    fig.tight_layout()
    fig.savefig(f'image/interpret/ale/{ft.replace("#","num")}_inter{model}.png')
    fig.show()
    plt.show()

pylab.rcParams['font.size'] = 5
for i, ft in enumerate(continuous_fts):
    fig = plt.figure(figsize=(2,1.4),dpi=500)
    axis = fig.add_subplot()
    ale_eff = ale(
        X=ale_inter_x_test, model=ale_func_inter, feature=[ft], grid_size=50, 
        feature_type='discrete' if ft in discrete_fts else 'continuous',
        include_CI=False, fig=fig, ax=axis
    ) # Keys: ['eff', 'size']
    fig.tight_layout()
    fig.savefig(f'image/interpret/ale/{ft.replace("#","num")}_inter{model}.png')
    fig.show()
    plt.show()

print('Finish')
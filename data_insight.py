import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
# from top2vec import Top2Vec
from sklearn.preprocessing import normalize
from utils import *
from models import *
from transformers import AutoTokenizer
import tensorflow_hub as hub
from torch_scatter import scatter
from torch import nn
import numpy as np
import language_tool_python
import math
import time
from numba import cuda 

args = parse_train_args()
args.device = 'cuda'if torch.cuda.is_available() else 'cpu'
device = cuda.get_current_device()
args.train_eval_sample = 'train'
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
embed_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
file = open(args.generated_data_file, "a")
index = args.chunk
dataset_lst = {'amazon_review_full':5,
               'amazon_review_polarity':2,'dbpedia':14,
               'yahoo_answers':10,'ag_news':4,
               'yelp_review_full':5,'yelp_review_polarity':2,
               'banking77__2':2, 'banking77__4':4, 'banking77__5':5, 
               'banking77__10':10, 'banking77__14':14,
               'tweet_eval_emoji_2':2, 'tweet_eval_emoji_4':4, 'tweet_eval_emoji_5':5, 
               'tweet_eval_emoji_10':10, 'tweet_eval_emoji_14':14,
              }

with open(f'generated_data/dataset_{index}.txt','r') as f:
    args.dataset = f.read()

test_index = np.load(f'generated_data/test_index_{index}.npy')
train_index = np.load(f'generated_data/train_index_{index}.npy')
train_data, test_data = load_dataset(args,train_index,test_index,args.custom_data) # Dataframe
documents = train_data['text'].tolist() # list
labels = train_data['label'].tolist() # list

tokens = []
snt_len = []
for text in documents:
    snt_tokens = tokenizer(text)['input_ids']
    tokens += snt_tokens
    snt_len.append(len(snt_tokens))

########### Embedding ###########
# embeddings = model._embed_documents(documents,32)
embeddings = normalize(embed_model(documents))
del embed_model
torch.cuda.empty_cache()
device.reset()

########### Class Separation ###########
means = scatter(torch.tensor(embeddings),torch.tensor(labels),dim=0,reduce='mean')
pdist = nn.PairwiseDistance(p=2)
inter_dst = []
for i in range(len(means)):
    dst_i = []
    for j in range(len(means)):
        dst_i.append(pdist(means[i],means[j]).item())
    inter_dst.append(dst_i)
glb_mean = np.mean(embeddings)
within_dst = pdist(torch.tensor(embeddings),means[torch.tensor(labels)])
between_dst = pdist(means[torch.tensor(labels)],torch.full(embeddings.shape,glb_mean))
n_labels = len(list(set(labels)))

# Mean inter-distance
mean_dst = 2*np.sum(np.array(inter_dst))/(len(inter_dst)*(len(inter_dst)-1))

# # Fisher Discriminant Ratio
between_var = torch.sum(between_dst)/(between_dst.shape[0]-1)
lda_ratio=torch.sum(within_dst)/torch.sum(between_dst)

# Calinski-Harabasz Index
cal_har_index = lda_ratio*(embeddings.shape[0]-n_labels)/(n_labels-1)

########### Clustering ###########
hdbscan_args = {'min_cluster_size': 5,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'}
cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(embeddings)
labels_clst = cluster.labels_
embeddings = np.array(embeddings)[np.array(labels_clst)>=0]
labels_clst = np.array(labels_clst)[np.array(labels_clst)>=0]
n_labels_clst = len(list(set(labels_clst)))

del cluster
torch.cuda.empty_cache()

# Davies-Bouldin Index
means = scatter(torch.tensor(embeddings),torch.tensor(labels_clst),dim=0,reduce='mean')
within_dst = pdist(torch.tensor(embeddings),means[torch.tensor(labels_clst)])
r_ij = []
avg_within_dst = scatter(within_dst,torch.tensor(labels_clst),dim=0,reduce='mean')
inter_dst = []
for i in range(len(means)):
    dst_i = []
    for j in range(len(means)):
        dst_i.append(pdist(means[i],means[j]).item())
    inter_dst.append(dst_i)
for i in range(n_labels_clst):
    r_i = []
    for j in range(n_labels_clst):
        if i != j:
            r_i += [(avg_within_dst[i]+avg_within_dst[j])/inter_dst[i][j]]
        else:
            r_i += [0]
    r_ij.append(r_i)
davies_bouldin_idx = sum([max(r_ij[i]) for i in range(len(r_ij))])/len(r_ij) if len(r_ij) > 0 else 0

########### Distribution of labels ###########
# In case n_class == 2 --> No skew
lb_dis = np.array([0 for i in range(dataset_lst[args.dataset])])
unique, counts = np.unique(labels, return_counts=True)
lb_dis[:counts.shape[0]] = counts
# lb_dis = lb_dis/np.sum(lb_dis)

# Pearson median skewness
skn = 3*(np.mean(lb_dis)-np.median(lb_dis))/np.std(lb_dis) if np.std(lb_dis) !=0 else 0

# Kurtosis
kts = np.mean((lb_dis-np.mean(lb_dis))**4)/np.std(lb_dis)**4 if np.std(lb_dis) !=0 else 0

# Misclassification rate
args.model = 'char_cnn'
args.train_eval_sample = 'train'
args.number_of_characters = len(args.alphabet)+len(args.extra_characters)
args.number_of_class = dataset_lst[args.dataset]
model, tokenizer = get_model(args)
train_data_tmp, test_data_tmp = preprocess_data(args, tokenizer, train_data, test_data) # Dataset
model = model.to(args.device)
train_loader = construct_loader(args, train_data_tmp)
test_loader = construct_loader(args, test_data_tmp)
train(args,train_loader,test_loader,model)
del test_data
torch.cuda.empty_cache()

args.train_eval_sample = 'eval'
test_error_index = np.load(f'generated_data/{args.dataset}_test_error_index.npy')
test_data = load_dataset(args,test_index=test_error_index)
clsf = get_clsf(args, model, tokenizer)
pred = np.array(clsf.get_pred(test_data['text'].tolist()).cpu())
miss_clsf_rate = 1-(test_data['label']==pred).sum()/pred.shape[0]
print(miss_clsf_rate)

# # Grammatical Error
# language_tool = language_tool_python.LanguageTool('en-US')
# print(len(documents))
# print(language_tool.check(documents))
# grammar_error = len(language_tool.check(documents))
# del language_tool
# torch.cuda.empty_cache()

# # Fluency
# for snt in documents:
#     flu_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#     lm = GPT2LMHeadModel.from_pretrained("gpt2")
#     ipt = flu_tokenizer(snt, return_tensors="pt", verbose=False)
#     fluency = math.exp(lm(**ipt, labels=ipt.input_ids)[0])
# file.write(f'{miss_clsf_rate}')
file.write(f'{index},{args.dataset},{sum(snt_len)/len(snt_len)},{len(list(set(tokens)))},{min(snt_len)},{max(snt_len)},{mean_dst},{lda_ratio},{cal_har_index},{davies_bouldin_idx},{n_labels_clst},{skn},{len(unique)},{kts},{miss_clsf_rate},{dataset_lst[args.dataset]},')
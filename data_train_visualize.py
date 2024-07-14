import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import matplotlib.pylab as pylab
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
import umap
import hdbscan

args = parse_train_args(visualize=True)
args.device = 'cuda'if torch.cuda.is_available() else 'cpu'
args.train_eval_sample = 'train'
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
embed_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
file = open("generated_data/data_test.csv", "a")
# index = args.chunk
# index=71
pylab.rcParams['font.size'] = 25
dataset_lst = {'amazon_review_full':5,
               'amazon_review_polarity':2,'dbpedia':14,
               'yahoo_answers':10,'ag_news':4,
               'yelp_review_full':5,'yelp_review_polarity':2,
               'banking77__2':2, 'banking77__4':4, 'banking77__5':5, 
               'banking77__10':10, 'banking77__14':14,
               'tweet_eval_emoji_2':2, 'tweet_eval_emoji_4':4, 'tweet_eval_emoji_5':5, 
               'tweet_eval_emoji_10':10, 'tweet_eval_emoji_14':14,
              }

for index in range(0,30):
    with open(f'generated_data/dataset_{index}.txt', 'r') as f:
        dataset = f.read()
    args.dataset, args.number_of_class = dataset, dataset_lst[dataset]
    test_index = np.load(f'generated_data/test_index_{index}.npy')
    train_index = np.load(f'generated_data/train_index_{index}.npy')
    train_data, test_data = load_dataset(args,train_index,test_index) # Dataframe
    documents = train_data['text'].tolist() # list
    labels = train_data['label'].tolist() # list

    tokens = []
    snt_len = []
    for text in documents:
        snt_tokens = tokenizer(text)['input_ids']
        tokens += snt_tokens
        snt_len.append(len(snt_tokens))

    ########### Number of tokens ###########
    plt.figure()
    plt.hist(snt_len,bins=100)
    # plt.suptitle(f'Distribution of the # of tokens in sentences in train data')
    plt.ylabel('# of sentences')
    plt.xlabel('# of tokens')
    plt.tight_layout()
    plt.savefig(f'image/num_tokens/{args.dataset}_{index}_token.png')
    plt.close()

    ########### Embedding ###########
    embeddings = normalize(embed_model(documents))
    umap_args = {'n_neighbors': 100,
                 'min_dist': 0.1,
                 'n_components': 2,
                 'metric': 'cosine'}
    hdbscan_args = {'min_cluster_size': 15,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}
    umap_model = umap.UMAP(**umap_args).fit(embeddings)
    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                  init='random', perplexity=3).fit_transform(model.document_vectors)
    cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)
    color = ['b','g','r','c','m','y','#d36500','#3c728e','#3ce18e','#ffa68e','#6aa68e','#0d86fc','#700d06','#709e06','k']
    plt.figure()
    for i in range(len(umap_model.embedding_)):
        if cluster.labels_[i] == -1:
            continue
        clr = color[cluster.labels_[i]]
        x,y = umap_model.embedding_[i]
        plt.plot(x,y,color=clr,marker='o')
    plt.tight_layout()
    plt.savefig(f'image/embedding/{args.dataset}_{index}_embedding.png')
    plt.close()

    ########### Distribution of labels ###########
    lb_dis = np.array([0 for i in range(dataset_lst[args.dataset])])
    unique, counts = np.unique(labels, return_counts=True)
    lb_dis[:counts.shape[0]] = counts
    plt.figure()
    plt.bar(range(lb_dis.shape[0]), lb_dis)
    # plt.plot(df['Year'], df['Sample Size'], '-o', color='orange')
    # plt.hist(labels)
    # plt.suptitle(f'Distribution of labels in train data')
    plt.ylabel('# of sentences')
    plt.xlabel('Label')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'image/lb_dis/{args.dataset}_{index}_label.png')
    plt.close()
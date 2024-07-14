import pandas as pd
from random import sample, shuffle

def load_dataset(args, train_index=None, test_index=None, custom_data=False):
    if args.train_eval_sample == 'train':
        assert train_index is not None and test_index is not None
    if args.train_eval_sample == 'eval':
        assert test_index is not None

    if args.dataset == 'imdb':
        import datasets
        train_data, test_data = datasets.load_dataset(args.dataset, split =['train','test'])
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.DataFrame({'text':train_data['text'], 
                                     'label':train_data['label']})
        else:
            df_test = pd.DataFrame({'text':test_data['text'], 
                                    'label':test_data['label']})
    elif args.dataset == 'ag_news':
        names = ['label','text','description'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/ag_news/train.csv', names=names, index_col=False)
        else:
            df_test = pd.read_csv('datasets/ag_news/test.csv', names=names, index_col=False)
    elif args.dataset == 'amazon_review_full':
        names = ['label','text','review_text'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/amazon_review_full/train.csv', names=names, index_col=False)
        else:
            df_test = pd.read_csv('datasets/amazon_review_full/test.csv', names=names, index_col=False)        
    elif args.dataset == 'amazon_review_polarity':
        names = ['label','text','review_text'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/amazon_review_polarity/train.csv', names=names, index_col=False)
        else:
            df_test = pd.read_csv('datasets/amazon_review_polarity/test.csv', names=names, index_col=False)
    elif args.dataset == 'dbpedia':
        names = ['label','text','content'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/dbpedia/train.csv', names=names, index_col=False)
        else:
            df_test = pd.read_csv('datasets/dbpedia/test.csv', names=names, index_col=False)
    elif args.dataset == 'sogou_news':
        names = ['label','text','content'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/sogou_news/train.csv', names=names, index_col=False)
        else:
            df_test = pd.read_csv('datasets/sogou_news/test.csv', names=names, index_col=False)
    elif args.dataset == 'yahoo_answers':
        names = ['label','text','content','best_answer'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text','best_answer']
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/yahoo_answers/train.csv', names=names, on_bad_lines='skip', index_col=False)
        else:
            df_test = pd.read_csv('datasets/yahoo_answers/test.csv', names=names, on_bad_lines='skip', index_col=False)
    elif args.dataset == 'yelp_review_full':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/yelp_review_full/train.csv', names=['label','text'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/yelp_review_full/test.csv', names=['label','text'], index_col=False)
    elif args.dataset == 'yelp_review_polarity':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/yelp_review_polarity/train.csv', names=['label','text'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/yelp_review_polarity/test.csv', names=['label','text'], index_col=False)
    elif args.dataset == 'banking77__2':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/banking77__2/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/banking77__2/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'banking77__4':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/banking77__4/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/banking77__4/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'banking77__5':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/banking77__5/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/banking77__5/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'banking77__10':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/banking77__10/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/banking77__10/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'banking77__14':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/banking77__14/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/banking77__14/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'tweet_eval_emoji_2':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/tweet_eval_emoji_2/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/tweet_eval_emoji_2/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'tweet_eval_emoji_4':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/tweet_eval_emoji_4/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/tweet_eval_emoji_4/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'tweet_eval_emoji_5':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/tweet_eval_emoji_5/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/tweet_eval_emoji_5/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'tweet_eval_emoji_10':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/tweet_eval_emoji_10/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/tweet_eval_emoji_10/test.csv', names=['text','label'], index_col=False)
    elif args.dataset == 'tweet_eval_emoji_14':
        if args.train_eval_sample in ['train','sample_train']:
            df_train = pd.read_csv('datasets/tweet_eval_emoji_14/train.csv', names=['text','label'], index_col=False)
        else:
            df_test = pd.read_csv('datasets/tweet_eval_emoji_14/test.csv', names=['text','label'], index_col=False)
    
    if args.train_eval_sample in ['train','sample_train']:
        df_train = df_train[df_train.notnull().all(1)]
        df_train['label'] = df_train['label'].astype('int64')
    else:
        df_test = df_test[df_test.notnull().all(1)]
        df_test['label'] = df_test['label'].astype('int64')

    if args.train_eval_sample == 'sample_train':
        index_lst = sample(range(len(df_train[['label','text']][df_train.notnull().all(1)].index)),args.limit_train+args.limit_test)
        shuffle(index_lst)
        return index_lst[:args.limit_train], index_lst[:args.limit_test]
    elif args.train_eval_sample == 'sample_attack':
        return sample(range(len(df_test[['label','text']][df_test.notnull().all(1)].index)),args.limit_test)
    elif args.train_eval_sample == 'train':
        data_train = df_train[['label','text']][df_train.notnull().all(1)].iloc[train_index].reset_index(drop=True)
        if custom_data:
            data_add = pd.read_csv(args.custom_data_dir,index_col=False)
            data_train = pd.concat([data_train, data_add]).reset_index(drop=True)
        return data_train,\
           df_train[['label','text']][df_train.notnull().all(1)].iloc[test_index].reset_index(drop=True)
    elif args.train_eval_sample == 'eval':
        return df_test[['label','text']][df_test.notnull().all(1)].iloc[test_index].reset_index(drop=True)
from sklearn import preprocessing    
from tqdm import tqdm
import pandas as pd
import torch
import datasets

def preprocess_data(args, tokenizer, train_data=None, test_data=None):

    assert args.train_eval_sample == 'train' and train_data is not None or \
           args.train_eval_sample == 'eval' and train_data is None
    tqdm.pandas()
    with open('temp_data.csv','a') as file:
        file.write(str(type(tokenizer)))
    # Process 'text'
    print('Processing text')
    
    def tokenization(dataset_text):
        if args.model == 'char_cnn':
            ret = tokenizer(dataset_text) # Deprecated
        else:
            ret = list(tokenizer(dataset_text, padding = 'max_length', truncation=True)['input_ids'])
        return ret
    
    if args.train_eval_sample == 'train':
        train_data['text'] = train_data['text'].progress_map(tokenization)
        test_data['text'] = test_data['text'].progress_map(tokenization)
    print('-'*50)

    # Process 'label'
    print('Processing label')
    if args.dataset != 'imdb':
        if args.train_eval_sample == 'train':
            train_data['label'] = train_data['label'].progress_map(lambda x:x-1)
        test_data['label'] = test_data['label'].progress_map(lambda x:x-1)
    print('-'*50)
   
    if args.train_eval_sample == 'train':
        train_data = datasets.Dataset.from_dict({'text':train_data['text'].tolist(), 'label':train_data['label'].tolist()})
        test_data = datasets.Dataset.from_dict({'text':test_data['text'].tolist(), 'label':test_data['label'].tolist()})
        return train_data, test_data    
    test_data = datasets.Dataset.from_dict({'x':test_data['text'].tolist(), 'y':test_data['label'].tolist()})
    return test_data

def preprocess_huggingface(args, tokenizer, train_data=None, test_data=None):
    assert args.train_eval_sample == 'train' and train_data is not None or \
           args.train_eval_sample == 'eval' and train_data is None
    
    # Process 'label'
    print('Processing label')
    if args.dataset != 'imdb':
        tqdm.pandas()
        if args.train_eval_sample == 'train':
            train_data['label'] = train_data['label'].progress_map(lambda x:x-1)
        test_data['label'] = test_data['label'].progress_map(lambda x:x-1)
    print('-'*50)
    if args.train_eval_sample == 'eval':
        return datasets.Dataset.from_dict({'x':test_data['text'].tolist(),'y':test_data['label'].tolist()})

    # Tokenize
    train_encodings = tokenizer(train_data['text'].tolist(), padding = 'max_length', truncation=True, return_tensors='pt')
    test_encodings = tokenizer(test_data['text'].tolist(), padding = 'max_length', truncation=True, return_tensors='pt')
    train_encodings = {key:value for key, value in train_encodings.items()}
    test_encodings = {key:value for key, value in test_encodings.items()}

    test_encodings['label'] = test_data['label'].tolist()
    train_encodings['label'] = train_data['label'].tolist()
    test_data = datasets.Dataset.from_dict(test_encodings)
    train_data = datasets.Dataset.from_dict(train_encodings)
    return train_data, test_data
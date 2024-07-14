import OpenAttack
import torch
from torch import nn
import numpy as np

class MyClassifier(OpenAttack.Classifier):
        def __init__(self, model,tokenizer,model_name,device):
            self.model = model
            self.model_name = model_name
            self.tokenizer = tokenizer
            self.device = device

        def get_pred(self, input_):
            if self.model_name in ['roberta-base','distilroberta-base',
                                   'bert-base','distilbert-base',
                                   'microsoft/deberta-base','google/electra-base-discriminator','gpt2','t5-base','facebook/bart-base']:
                input_sent = self.tokenizer(input_, padding = 'max_length', truncation=True, return_tensors='pt')
                input_sent = {key:value.to(self.device) for key, value in input_sent.items()}
                res = self.model(**input_sent).logits.argmax(axis=1) 
            elif self.model_name == 'char_cnn':
                input_sent = self.tokenizer(input_)
                input_sent = torch.tensor(input_sent, device=self.device, dtype=torch.float)
                if len(input_sent.shape) == 2:
                    input_sent = input_sent.unsqueeze(0)
                res = self.model(input_sent).argmax(axis=1)
            else:
                input_ = self.tokenizer(input_, padding = 'max_length', truncation=True)
                input_ = torch.tensor(input_['input_ids'], device=self.device, dtype=torch.float)
                res = self.model(input_).argmax(axis=1)
            return res

        def get_prob(self, input_):
            ret = []
            for sent in input_:
                if self.model_name in ['roberta-base','distilroberta-base',
                                       'bert-base','distilbert-base',
                                       'microsoft/deberta-base','google/electra-base-discriminator','gpt2','facebook/bart-base']:
                    input_sent = self.tokenizer(sent, padding = 'max_length', truncation=True, return_tensors='pt')
                    input_sent = {key:value.to(self.device) for key, value in input_sent.items()}
                    res = self.model(**input_sent).logits
                elif self.model_name == 'char_cnn':
                    input_sent = self.tokenizer(sent)
                    input_sent = torch.tensor(input_sent, device=self.device, dtype=torch.float).unsqueeze(0)
                    res = self.model(input_sent)
                else:
                    input_sent = self.tokenizer(sent, padding = 'max_length', truncation=True)
                    input_sent = torch.tensor([input_sent['input_ids']], device=self.device, dtype=torch.float)
                    res = self.model(input_sent)
                prob = nn.Softmax(dim=1)(res).squeeze()
                ret.append(prob.cpu().detach().numpy())
            return np.array(ret)

def get_clsf(args, model, tokenizer): 
    return MyClassifier(model, tokenizer, args.model, args.device)
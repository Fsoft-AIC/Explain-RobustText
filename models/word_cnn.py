from torch import nn
import torch
import torch.nn.functional as F

# This code is referenced from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
class WordLevelCNN(nn.Module):
    def __init__(self,args):
        super(WordLevelCNN, self).__init__()
        
        V = args.vocab_size
        D = args.embed_dim
        C = args.number_of_class
        Ci = 1
        Co = args.kernel_num
        Ks = [int(k) for k in args.kernel_sizes.split(',')]

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = x.type(torch.long)
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
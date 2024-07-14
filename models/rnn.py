import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
	def __init__(self,args):
		super(RNN, self).__init__()

		self.batch_size = args.batch_size
		self.hidden_dim = args.embed_dim
		self.stacked_layers = args.stacked_layers
		self.vocab_size = args.vocab_size
		
		self.model = args.model
		self.device = args.device

		self.dropout = nn.Dropout(0.5)
		self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
		if args.model == 'rnn':
			self.lstm = nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.stacked_layers, batch_first=True)
			self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
			self.fc2 = nn.Linear(self.hidden_dim, args.number_of_class)
		else:
			self.lstm = nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.stacked_layers, bidirectional=True, batch_first=True)
			self.fc1 = nn.Linear(in_features=self.hidden_dim*2, out_features=self.hidden_dim)
			self.fc2 = nn.Linear(self.hidden_dim, args.number_of_class)
			
	def forward(self, x):

		if self.model == 'rnn':
			h = torch.zeros((self.stacked_layers, x.size(0), self.hidden_dim), device=self.device)
		else:
			h = torch.zeros((self.stacked_layers*2, x.size(0), self.hidden_dim), device=self.device)
		
		torch.nn.init.xavier_normal_(h)

		x = x.type(torch.long)
		out = self.embedding(x)
		out, hidden = self.lstm(out, h)
		out = self.dropout(out)
		out = torch.relu_(self.fc1(out[:,-1,:]))
		out = self.dropout(out)
		out = self.fc2(out)
		
		return out
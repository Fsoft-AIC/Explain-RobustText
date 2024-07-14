import json
import torch
import torch.nn as nn
import numpy as np

# This code is referenced from https://github.com/ahmedbesbes/character-based-cnn/blob/593197610498bf0b4898b3bdf2e1f6730f954613/src/model.py

class CharacterLevelCNN(nn.Module):
    def __init__(self, args):
        super(CharacterLevelCNN, self).__init__()

        # define conv layers
        self.dropout_input = nn.Dropout2d(args.dropout_input)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.number_of_characters,
                args.max_length_char_cnn,
                kernel_size=args.kernel_size[0],
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(args.max_pool),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(args.max_length_char_cnn, args.max_length_char_cnn, kernel_size=args.kernel_size[0], padding=0), nn.ReLU(), nn.MaxPool1d(args.max_pool)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(args.max_length_char_cnn, args.max_length_char_cnn, kernel_size=args.kernel_size[1], padding=0), nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(args.max_length_char_cnn, args.max_length_char_cnn, kernel_size=args.kernel_size[1], padding=0), nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(args.max_length_char_cnn, args.max_length_char_cnn, kernel_size=args.kernel_size[1], padding=0), nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(args.max_length_char_cnn, args.max_length_char_cnn, kernel_size=args.kernel_size[1], padding=0), nn.ReLU(), nn.MaxPool1d(args.max_pool)
        )

        # compute the  output shape after forwarding an input to the conv layers

        input_shape = (args.batch_size,args.max_length_char_cnn,args.number_of_characters)
        self.output_dimension = self._get_conv_output(input_shape)

        # define linear layers

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, args.max_length_char_cnn*2), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(nn.Linear(args.max_length_char_cnn*2, args.max_length_char_cnn*2), nn.ReLU(), nn.Dropout(0.5))

        self.fc3 = nn.Linear(args.max_length_char_cnn*2, args.number_of_class)

        # initialize weights

        self._create_weights()

    # utility private functions

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    # forward

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CharCNNTokenizer():
    def __init__(self, args):
        self.vocabulary = args.alphabet + args.extra_characters
        self.number_of_characters = args.number_of_characters
        self.max_length_char_cnn = args.max_length_char_cnn
        self.identity_mat = np.identity(self.number_of_characters)
    
    def __call__(self, inputs):
        if type(inputs) is str:
            return self.tkn_str(inputs)
        elif type(inputs) is list:
            ret = []
            for input_str in inputs:
                ret.append(self.tkn_str(input_str))
            return ret
        else:
            raise Exception('Ambiguous type') 

    def tkn_str(self, raw_text):
        assert type(raw_text) is str

        data = np.array(
            [
                self.identity_mat[self.vocabulary.index(char)]
                for char in list(raw_text.lower())[::-1]
                if char in self.vocabulary
            ],
            dtype=np.float32,
        )

        ret = np.array(
            [
                [0 for j in range(self.number_of_characters)] 
                for i in range(self.max_length_char_cnn)
            ], 
            dtype=np.float32
        )
        if len(data.shape) == 2:
            ret[:data.shape[0]] = data[:ret.shape[0]]
        
        return ret.tolist()


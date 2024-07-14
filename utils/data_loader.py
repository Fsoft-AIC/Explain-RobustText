from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, args, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return input_ids, label

# class MyHuggingFaceDataset(Dataset):
#     def __init__(self, args, dataset):
#         self.dataset = dataset
#         self.length = len(dataset.index)
#         self.device = args.device

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         data = {key: torch.tensor(value) for key, value in self.dataset.iloc[index].items()}
#         return data

# def construct_dataset(args, dataset):
#     huggingface_dataset = MyHuggingFaceDataset(args, dataset)
#     return huggingface_dataset

def construct_loader(args, dataset):
    input_ids = np.array(dataset['text'])
    labels = np.array(dataset['label'])
    data = MyDataset(args, input_ids, labels)
    return DataLoader(dataset=data,
                      batch_size=args.batch_size,
                      pin_memory=False,
                      shuffle=True)
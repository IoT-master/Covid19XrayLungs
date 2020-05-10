# Idea was inspired by:
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import torch
from torch.utils import data
import pandas as pd

class DataSpliter(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataframe, features, labels):
        'Initialization'
        super(DataSpliter, self).__init__()
        self.dataframe = dataframe
        self.labels = labels
        self.list_IDs = features

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.FloatTensor(self.dataframe[self.list_IDs].to_numpy())[index]
        y = (torch.LongTensor(self.dataframe[self.labels].to_numpy())).squeeze(1)[index]

        return X, y
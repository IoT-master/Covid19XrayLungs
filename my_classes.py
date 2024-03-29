# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

import torch
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self):
        'Initialization'

  def __len__(self):
        'Denotes the total number of samples'
        return len(shape)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

if __name__=='__main__':
    pass
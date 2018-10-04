import os
import torch
import torch.utils.data as data
import pickle
import numpy as np


class AwA(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()

        path = opt.data_dir + '/AwA/pkls/'
        with open(path + split + '.pkl', 'rb') as f:
            data = pickle.load(f)

        with open(path + 'sem_embed.pkl', 'rb') as f:
            S = pickle.load(f)

        with open(path + 'ids.pkl', 'rb') as f:
            ids = pickle.load(f)

        self.X = torch.tensor(data['X'].astype(np.float32))
        self.Y = torch.tensor(data['Y'].astype(np.long))
        self.S = torch.tensor(S.astype(np.float32))
        self.ids = torch.tensor(ids[split].astype(np.long))
        self.vdim = self.X.shape[1]
        self.sdim = self.S.shape[1]
        self.n_classes = self.S.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'Y': self.Y[idx]
        }

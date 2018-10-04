import os
import torch
import numpy as np
from easydict import EasyDict as edict
import yaml
import pickle
import argparse


class Options(object):
    def __init__(self):

        opt = edict()
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--exp_dir', type=str)
        parser.add_argument('--exp_name', type=str)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--which_epoch', type=str)
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--model', type=str)
        parser.add_argument('--isTrain', action='store_true')
        parser.add_argument('--gpu_id', type=int, default=-1)
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--lr_decay_step', type=int, default=1)
        parser.add_argument('--lr_decay_rate', type=float, default=0.9)
        parser.add_argument('--max_grad_value', type=float, default=5.0)
        parser.add_argument('--max_grad_norm', type=float, default=100.)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--save_freq', type=int)
        parser.add_argument('--print_freq', type=int)
        #################################################
        parser.add_argument('--vdim', type=int)
        parser.add_argument('--sdim', type=int)
        parser.add_argument('--zdim', type=int)
        parser.add_argument('--n_classes', type=int)
        parser.add_argument('--using_all_classes', action='store_true')
        parser.add_argument('--using_classification_loss', action='store_true')
        parser.add_argument('--using_rec_loss', action='store_true')
        parser.add_argument('--using_kl_loss', action='store_true')
        parser.add_argument('--using_margin_loss', action='store_true')
        parser.add_argument('--lambda_rec', type=float)
        parser.add_argument('--lambda_kl', type=float)
        parser.add_argument('--lambda_margin', type=float)
        parser.add_argument('--lambda_cls', type=float)
        #################################################
        self.parser = parser

    def __print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        with open(os.path.join(self.opt.exp_dir, 'opt.txt'), 'w') as f:
            f.write(message)
            f.write('\n')

    def parse(self):
        self.opt = self.parser.parse_args()

        if self.opt.seed < 0:
            self.opt.seed = np.random.randint(0, 1000)

        if self.opt.gpu_id >= 0:
            self.opt.device = torch.device('cuda:{}'.format(self.opt.gpu_id))
        else:
            self.opt.device = torch.device('cpu')

        self.opt.exp_dir = os.path.join(self.opt.exp_dir, self.opt.exp_name)
        if os.path.exists(self.opt.exp_dir):
            print('Warning: folder exists!!!')
        else:
            os.makedirs(self.opt.exp_dir)

        self.__print_options()

        return self.opt

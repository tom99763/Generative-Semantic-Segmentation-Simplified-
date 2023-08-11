import torch
import torch.nn as nn
from torch import optim
import numpy as np
import sys
sys.path.append('./models')
from lr_schedule import *
from losses import *
import pandas as pd
from experiments import *
from decode_heads import *
import os

class Trainer(nn.Module):
    def __init__(self, config, opt):
        super().__init__()
        self.config= config
        self.opt = opt
        self.M = MaskigeTT(config)
        self.I = GenerativeSegHead(config)

        #optimizer
        self.opt_M = optim.Adam(self.M.parameters(), **(config['optim']['M_params']))
        self.opt_I = optim.Adam(self.I.parameters(), **(config['optim']['I_params']))

        # record
        self.hist_train = {'i_iter': [], 'l_cls': []}
        self.hist_test = {'i_iter': []}

        if opt.mode == 'train_prior':
            for param in self.M.parameters():
                param.requires_grad = False
            for param in self.I.parameters():
                param.requires_grad = True

        if opt.mode == 'train_posterior':
            for param in self.I.parameters():
                param.requires_grad = False
            for param in self.M.parameters():
                param.requires_grad = True

    def train_posterior(self, x_s, x_t, y_s, y_t, i_iter):
        adjust_learning_rate(self.opt_M, self.config['optim']['M_params']['lr'], i_iter, self.opt.num_iters)
        self.opt_M.zero_grad()
        p_s = self.M.forward_maskige(y_s)
        l_cls = criterion(p_s, y_s)
        loss = l_cls
        loss.backward()
        self.opt_M.step()
        self.hist_train['i_iter'].append(i_iter)
        self.hist_train['l_cls'].append(l_cls.item())
        return l_cls

    def train_prior(self, x_s, x_t, y_s, y_t, i_iter):
        adjust_learning_rate(self.opt_I, self.config['optim']['I_params']['lr'], i_iter, self.opt.num_iters)
        self.opt_M.zero_grad()
        self.opt_I.zero_grad()
        feat_idx = self.M.get_feat_idx(y_s) #(b, h, w)
        logits = self.I(x_s) #(b, 8192, h, w)
        l_cls = criterion_latent(logits, feat_idx)
        loss = l_cls
        loss.backward()
        self.opt_I.step()
        self.hist_train['i_iter'].append(i_iter)
        self.hist_train['l_cls'].append(l_cls.item())
        return l_cls

    def test_step(self, loader, i_iter):
        pass

    def save_weights(self):
        if self.opt.mode == 'train_posterior':
            save_name = self.config['ckpt_dir_post']
        elif self.opt.mode == 'train_prior':
            save_name = self.config['ckpt_dir_prior']
        torch.save({
                'M': self.M.state_dict(),
                'I': self.I.state_dict(),
            }, save_name)

    def load_weights(self):
        if self.opt.mode == 'train_posterior':
            save_name = self.config['ckpt_dir_post']
        elif self.opt.mode == 'train_prior':
            save_name = self.config['ckpt_dir_prior']
            if os.path.exists(save_name) == False:
                print('no prior weights')
                save_name = self.config['ckpt_dir_post']
        ckpt_dir = torch.load(save_name)
        self.M.load_state_dict(ckpt_dir['M'])
        self.I.load_state_dict(ckpt_dir['I'])

    def save_hist(self):
        df = pd.DataFrame(
            np.stack([self.hist_train['i_iter'],
                      self.hist_train['l_cls'],
                      ], axis=0).T,
            columns=self.hist_train.keys()
        )
        df.to_csv(f"{self.config['output_dir']}/stats/hist_train_{self.opt.mode}.csv")

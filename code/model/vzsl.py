import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl_divergence
import itertools
from collections import OrderedDict


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, bias=bias)
        # init
        self.weight.data.uniform_(-1e-5, 1e-5)

    def forward(self, input):
        output = super().forward(input)
        return output


class MLP(nn.Module):
    def __init__(self, cin, hids, cout, dropout=0.0):
        super().__init__()

        layers = []
        for h in hids:
            layers.append(nn.Linear(cin, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            cin = h
        layers.append(LinearZeros(cin, cout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VZSL(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.encoder_net = MLP(opt.vdim, [1000, 1000], opt.zdim * 2, 0.8)
        self.decoder_net = MLP(opt.zdim, [1000, 1000], opt.vdim, 0.8)
        self.prior_net = LinearZeros(opt.sdim, opt.zdim * 2, bias=False)
        if opt.using_classification_loss:
            self.classifier = nn.Linear(opt.n_classes, opt.n_classes)

        if opt.isTrain:
            params = []
            params.append(self.encoder_net.parameters())
            params.append(self.decoder_net.parameters())
            params.append(self.prior_net.parameters())
            if opt.using_classification_loss:
                params.append(self.classifier.parameters())
            all_params = itertools.chain(*params)
            self.optimizer = torch.optim.Adam(all_params, lr=0.001)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)
        else:
            self.load(opt.which_epoch)

        self.encoder_net.to(opt.device)
        self.decoder_net.to(opt.device)
        self.prior_net.to(opt.device)
        if opt.using_classification_loss:
            self.classifier.to(opt.device)

    def set_input(self, input):
        '''
        Args:
                S: semantic embedding for all K classes [K, sdim]
                X: visual embedding [N, vdim]
                Y: class id [N]
        '''
        if 'ids' in input.keys():
            self.ids = input['ids'].to(self.opt.device)
        if 'S' in input.keys():
            self.S = input['S'].to(self.opt.device)
        if 'X' in input.keys():
            self.X = input['X'].to(self.opt.device)
        if 'Y' in input.keys():
            self.Y = input['Y'].to(self.opt.device)

    def encode(self, x):
        out = self.encoder_net(x)
        mean, logs = out[:, :self.opt.zdim], out[:, self.opt.zdim:]

        return mean, logs

    def reparameterize(self, mean, logs):
        if self.training:
            std = torch.exp(logs)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        else:
            return mean

    def decode(self, z):
        return self.decoder_net(z)

    def prior(self, s):
        out = self.prior_net(s)
        mean, logs = out[:, :self.opt.zdim], out[:, self.opt.zdim:]

        return mean, logs

    def __rec_loss(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=-1)

    def __kl(self, m1, s1, m2, s2):
        '''
        [zdim]
        '''
        return torch.sum(s2 - s1 + (torch.exp(2 * s1) + (m1 - m2)**2) /
                         2 / torch.exp(2 * s2) - 0.5)

    def __pairwise_kl(self, m1, s1, m2, s2):
        '''
        Args:
            m1, s1: [N, zdim]
            m2, s2: [K, zdim]
        Return:
            kl: [N, K]
        '''
        m1, s1 = m1.unsqueeze(1), s1.unsqueeze(1)
        return torch.sum(s2 - s1 + (torch.exp(2 * s1) + (m1 - m2)**2) /
                         2 / torch.exp(2 * s2) - 0.5, dim=-1)

    def __kl_loss(self, m1, s1, m2, s2):
        '''
        m1, s1, m2, s2: [N, zdim]
        '''
        return torch.sum(s2 - s1 + (torch.exp(2 * s1) + (m1 - m2)**2) /
                         2 / torch.exp(2 * s2) - 0.5, dim=-1)

    def __margin_loss(self, m1, s1, m2, s2):
        '''
        Args:
            m1, s1: [N, zdim]
            m2, s2: [K, zdim]
        Return:
            first compute pairwise KL divergence[N, K]
            then sum the last dim
            return [N]
        '''
        pass

    def forward(self):
        xmean, xlogs = self.encode(self.X)  # [N, zdim] [N, zdim]
        pmean, plogs = self.prior(self.S)  # [K, zdim] [K, zdim]
        z = self.reparameterize(xmean, xlogs)
        x_hat = self.decode(z)  # [N, vdim]
        loss = 0.0
        if self.opt.using_rec_loss:
            self.rec_loss = self.__rec_loss(self.X, x_hat)  # [N]
            loss = loss + self.rec_loss * self.opt.lambda_rec
        if self.opt.using_kl_loss:
            self.kl_loss = self.__kl_loss(
                xmean, xlogs, pmean[self.Y], plogs[self.Y])  # [N]
            loss = loss + self.kl_loss * self.opt.lambda_kl
        pair_kl = self.__pairwise_kl(xmean, xlogs, pmean, plogs)  # [N, K]

        if self.opt.using_all_classes:
            self.kl_logits = pair_kl
            self.kl_predict = torch.argmin(pair_kl, dim=1)
            self.kl_correct = self.kl_predict == self.Y
            if self.opt.using_margin_loss:
                self.margin_loss = torch.logsumexp(
                    -1.0 * pair_kl, dim=1)  # [N]
                loss = loss + self.margin_loss * self.opt.lambda_margin
            if self.opt.using_classification_loss:
                logits = self.classifier(-1.0 * pair_kl)
                self.cls_loss = F.cross_entropy(
                    logits, self.Y, reduction='none')  # [N]
                loss = loss + self.cls_loss * self.opt.lambda_cls
                self.cls_logits = logits
                self.cls_predict = torch.argmax(logits, dim=1)
                self.cls_correct = self.cls_predict == self.Y
        else:
            self.kl_logits = pair_kl[:, self.ids]
            self.kl_predict = self.ids[torch.argmin(self.kl_logits, dim=1)]
            self.kl_correct = self.kl_predict == self.Y
            if self.opt.using_margin_loss:
                self.margin_loss = torch.logsumexp(
                    -1.0 * self.kl_logits, dim=1)  # [N]
                loss = loss + self.margin_loss * self.opt.lambda_margin
            if self.opt.using_classification_loss:
                raise Exception()

        self.loss = loss.mean()

    def set_state(self, training):
        self.encoder_net.train(training)
        self.decoder_net.train(training)
        self.prior_net.train(training)
        if self.opt.using_classification_loss:
            self.classifier.train(training)

    def optimize(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss.backward()
        # clip grad
        if self.opt.max_grad_value > 0:
            nn.utils.clip_grad_value_(
                self.optimizer.param_groups[0]['params'], self.opt.max_grad_value)
        if self.opt.max_grad_norm > 0:
            total_norm = nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'], self.opt.max_grad_norm)
        self.optimizer.step()

    def train(self):
        self.set_state(True)
        self.optimize()

    def test(self):
        self.set_state(False)
        with torch.no_grad():
            self.forward()

    def get_output(self):
        out = OrderedDict()
        out['loss'] = self.loss.item()

        if self.opt.using_rec_loss:
            out['rec_loss'] = self.rec_loss.detach().cpu().numpy()

        if self.opt.using_kl_loss:
            out['kl_loss'] = self.kl_loss.detach().cpu().numpy()

        if self.opt.using_margin_loss:
            out['margin_loss'] = self.margin_loss.detach().cpu().numpy()

        if self.opt.using_classification_loss:
            out['cls_loss'] = self.cls_loss.detach().cpu().numpy()
            out['cls_logits'] = self.cls_logits.detach().cpu().numpy()
            out['cls_predict'] = self.cls_predict.detach().cpu().numpy()
            out['cls_correct'] = self.cls_correct.detach().cpu().numpy()

        out['kl_logits'] = self.kl_logits.detach().cpu().numpy()
        out['kl_predict'] = self.kl_predict.detach().cpu().numpy()
        out['kl_correct'] = self.kl_correct.detach().cpu().numpy()

        return out

    def save(self, which_epoch):
        save_path = os.path.join(
            self.opt.exp_dir, 'epoch_{}.pth'.format(which_epoch))
        state_dict = {
            'encoder_net': self.encoder_net.cpu().state_dict(),
            'decoder_net': self.decoder_net.cpu().state_dict(),
            'prior_net': self.prior_net.cpu().state_dict()
        }
        if self.opt.using_classification_loss:
            state_dict.update(
                {'classifier': self.classifier.cpu().state_dict()})
            self.classifier.to(self.opt.device)
        torch.save(state_dict, save_path)
        self.encoder_net.to(self.opt.device)
        self.decoder_net.to(self.opt.device)
        self.prior_net.to(self.opt.device)

    def load(self, which_epoch):
        load_path = os.path.join(
            self.opt.exp_dir, 'epoch_{}.pth'.format(which_epoch))
        state_dict = torch.load(load_path)
        self.encoder_net.load_state_dict(state_dict['encoder_net'])
        self.decoder_net.load_state_dict(state_dict['decoder_net'])
        self.prior_net.load_state_dict(state_dict['prior_net'])
        if self.opt.using_classification_loss:
            self.classifier.load_state_dict(state_dict['classifier'])

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate: %.7f' % lr)

import os
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from dataset import create_dataset
from model import create_model
from option.options import Options

opt = Options().parse()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

trainset = create_dataset(opt, 'train')
valset = create_dataset(opt, 'val')
testset = create_dataset(opt, 'test')
print('size of dataset => train:{} val:{} test:{}'.format(
    len(trainset), len(valset), len(testset)), flush=True)
trainloader = DataLoader(trainset, batch_size=opt.batch_size,
                         shuffle=True, num_workers=1)
valloader = DataLoader(valset, batch_size=opt.batch_size,
                       shuffle=False, num_workers=1)
testloader = DataLoader(testset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=1)

model = create_model(opt)


def cum_results(out, res):
    out['loss'] += res['loss']
    if 'rec_loss' in res.keys():
        out['rec_loss'] += res['rec_loss'].mean()
    if 'kl_loss' in res.keys():
        out['kl_loss'] += res['kl_loss'].mean()
    if 'margin_loss' in res.keys():
        out['margin_loss'] += res['margin_loss'].mean()
    if 'cls_loss' in res.keys():
        out['cls_loss'] += res['cls_loss'].mean()
    out['batch'] += 1.
    out['total'] += res['kl_correct'].size
    out['kl_correct'] += res['kl_correct'].sum()
    if 'cls_correct' in res.keys():
        out['cls_correct'] += res['cls_correct'].sum()

    return out


def avg_results(out):
    for k in out.keys():
        if 'loss' in k:
            out[k] /= out['batch']
        elif 'correct' in k:
            out[k] /= out['total']

    return out


def process_results(res):
    message = ''
    message += 'loss:{:.2f} '.format(res['loss'])
    if 'rec_loss' in res.keys():
        message += 'rec_loss:{:.2f} '.format(res['rec_loss'] * opt.lambda_rec)
    if 'kl_loss' in res.keys():
        message += 'kl_loss:{:.2f} '.format(res['kl_loss'] * opt.lambda_kl)
    if 'margin_loss' in res.keys():
        message += 'margin_loss:{:.2f} '.format(
            res['margin_loss'] * opt.lambda_margin)
    if 'cls_loss' in res.keys():
        message += 'cls_loss:{:.2f} '.format(res['cls_loss'] * opt.lambda_cls)
    if 'kl_correct' in res.keys():
        message += 'kl_acc:{:.2f} '.format(res['kl_correct'])
    if 'cls_correct' in res.keys():
        message += 'cls_acc:{:.2f} '.format(res['cls_correct'])

    return message


def train():
    out = defaultdict(float)
    data = {'S': trainset.S, 'ids': trainset.ids}
    model.set_input(data)
    for i, data in enumerate(trainloader):
        model.set_input(data)
        model.train()
        results = model.get_output()
        out = cum_results(out, results)

    return avg_results(out)


def val():
    out = defaultdict(float)
    data = {'S': valset.S, 'ids': valset.ids}
    model.set_input(data)
    for i, data in enumerate(valloader):
        model.set_input(data)
        model.test()
        results = model.get_output()
        out = cum_results(out, results)

    return avg_results(out)


def test():
    out = defaultdict(float)
    data = {'S': testset.S, 'ids': testset.ids}
    model.set_input(data)
    for i, data in enumerate(testloader):
        model.set_input(data)
        model.test()
        results = model.get_output()
        out = cum_results(out, results)

    return avg_results(out)


best_loss = np.inf
for epoch in range(opt.epochs):
    train_res = train()
    val_res = val()
    test_res = test()

    print('-' * 30, flush=True)
    print('Epoch:{}/{}'.format(epoch, opt.epochs))
    print('train => {}'.format(process_results(train_res)))
    print('val   => {}'.format(process_results(val_res)))
    print('test  => {}'.format(process_results(test_res)))

    if val_res['loss'] < best_loss:
        best_loss = val_res['loss']
        model.save('best')

    if opt.save_freq > 0 and (epoch + 1) % opt.save_freq == 0:
        model.save(epoch)

    model.update_learning_rate()

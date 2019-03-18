#!/usr/bin/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
#


import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import faiss


################################################

def LoadDataNLI(fn1, fn2, fn_lbl,
                dim=1024, bsize=32,
                fraction=1.0,
                shuffle=False, quiet=False):
    x = np.fromfile(fn1, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)
    faiss.normalize_L2(x)

    y = np.fromfile(fn2, dtype=np.float32, count=-1)
    y.resize(y.shape[0] // dim, dim)
    faiss.normalize_L2(y)

    lbl = np.loadtxt(fn_lbl, dtype=np.int32)
    lbl.reshape(lbl.shape[0], 1)

    if not quiet:
        print(' - read {:d}x{:d} elements in {:s}'.format(x.shape[0], x.shape[1], fn1))
        print(' - read {:d}x{:d} elements in {:s}'.format(y.shape[0], y.shape[1], fn2))
        print(' - read {:d} labels [{:d},{:d}] in {:s}'
              .format(lbl.shape[0], lbl.min(), lbl.max(), fn_lbl))

    if fraction < 1.0:
        N = int(x.shape[0] * fraction)
        if not quiet:
            print(' - using only the first {:d} examples'.format(N))
        x = x[:N][:]
        y = y[:N][:]
        lbl = lbl[:N][:]

    if not quiet:
        print(' - combine premises and hyps')
    nli = np.concatenate((x, y, np.absolute(x - y), np.multiply(x, y)), axis=1)

    D = data_utils.TensorDataset(torch.from_numpy(nli), torch.from_numpy(lbl))
    loader = data_utils.DataLoader(D, batch_size=bsize, shuffle=shuffle)
    return loader


################################################

class Net(nn.Module):
    def __init__(self, fname='',
                 idim=4*1024, odim=2, nhid=None,
                 dropout=0.0, gpu=0, activation='TANH'):
        super(Net, self).__init__()
        self.gpu = gpu
        if os.path.isfile(fname):
            print(' - loading mlp from %s'.format(fname))
            loaded = torch.load(fname)
            self.mlp = loaded.mlp
        else:
            modules = []
            print(' - mlp {:d}'.format(idim), end='')
            if len(nhid) > 0:
                if dropout > 0:
                    modules.append(nn.Dropout(p=dropout))
                nprev = idim
                for nh in nhid:
                    if nh > 0:
                        modules.append(nn.Linear(nprev, nh))
                        nprev = nh
                        if activation == 'TANH':
                            modules.append(nn.Tanh())
                            print('-{:d}t'.format(nh), end='')
                        elif activation == 'RELU':
                            modules.append(nn.ReLU())
                            print('-{:d}r'.format(nh), end='')
                        else:
                            raise Exception('Unrecognised activation {activation}')
                        if dropout > 0:
                            modules.append(nn.Dropout(p=dropout))
                modules.append(nn.Linear(nprev, odim))
                print('-{:d}, dropout={:.1f}'.format(odim, dropout))
            else:
                modules.append(nn.Linear(idim, odim))
                print(' - mlp {:d}-{:d}'.format(idim, odim))
            self.mlp = nn.Sequential(*modules)

        if self.gpu >= 0:
            self.mlp = self.mlp.cuda()

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dset, name=' Dev', nlbl=3, out_fname=None):
        correct = 0
        total = 0
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)
        if out_fname:
            fp = open(out_fname, 'w')
            fp.write('# outputs target_class predicted_class\n')
        for data in dset:
            X, Y = data
            Y = Y.long()
            if self.gpu >= 0:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).int().sum()
            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()
            if out_fname:
                for b in range(outputs.shape[0]):
                    for i in range(nlbl):
                        fp.write('{:f} '.format(outputs[b][i]))
                    fp.write('{:d} {:d}\n'
                             .format(predicted[b], Y[b]))

        print(' | {:4s}: {:5.2f}%'
              .format(name, 100.0 * correct.float() / total), end='')
        # print(' | loss {:6.4f}'.format(loss/total), end='')
        print(' | classes:', end='')
        for i in range(nlbl):
            print(' {:5.2f}'.format(100.0 * corr[i] / total), end='')

        if out_fname:
            fp.close()

        return correct, total


################################################

parser = argparse.ArgumentParser(
           formatter_class=argparse.RawDescriptionHelpFormatter,
           description='Classifier for NLI')

# Data
parser.add_argument(
    '--base-dir', '-b', type=str, required=True, metavar='PATH',
    help='Directory with all the data files)')
parser.add_argument(
    '--load', '-l', type=str, required=False, metavar='PATH', default='',
    help='Load network from file before training or for testing')
parser.add_argument(
    '--save', '-s', type=str, required=False, metavar='PATH', default='',
    help='File in which to save best network')
parser.add_argument(
    '--train', '-t', type=str, required=True, metavar='STR',
    help='Name of training corpus')
parser.add_argument(
    '--train-labels', '-T', type=str, required=True, metavar='STR',
    help='Name of training corpus (labels)')
parser.add_argument(
    '--dev', '-d', type=str, required=True, metavar='STR',
    help='Name of development corpus')
parser.add_argument(
    '--dev-labels', '-D', type=str, required=True, metavar='STR',
    help='Name of development corpus (labels)')
parser.add_argument(
    '--test', '-e', type=str, default=None,
    help='Name of test corpus without language extension')
parser.add_argument(
    '--test-labels', '-E', type=str, default=None,
    help='Name of test corpus without language extension (labels)')
parser.add_argument(
    '--lang', '-L', nargs='+', default=None,
    help='List of languages to test on')
parser.add_argument(
    '--cross-lingual', '-x', action='store_true',
    help='Also test on premise and hypothesis in different languages)')
parser.add_argument(
    '--parts', '-p', type=str, nargs='+', default=['prem', 'hyp'],
    help='Name of the two input parts to compare')
parser.add_argument(
    '--fraction', '-f', type=float, default=1.0,
    help='Fraction of training examples to use (from the beginning)')
parser.add_argument(
    '--save-outputs', type=str, default=None,
    help='File name to save classifier outputs ("l1-l2.txt" will be added)')

# network definition
parser.add_argument(
    '--dim', '-m', type=int, default=1024,
    help='dimension of sentence embeddings')
parser.add_argument(
    '--nhid', '-n', type=int, default=0, nargs='+',
    help='List of hidden layer(s) dimensions')
parser.add_argument(
    '--dropout', '-o', type=float, default=0.0, metavar='FLOAT',
    help='Value  of dropout')
parser.add_argument(
    '--nepoch', '-N', type=int, default=100, metavar='INT',
    help='Number of epochs')
parser.add_argument(
    '--bsize', '-B', type=int, default=128, metavar='INT',
    help='Batch size')
parser.add_argument(
    '--seed', '-S', type=int, default=123456789, metavar='INT',
    help='Initial random seed')
parser.add_argument(
    '--lr', type=float, default=0.001, metavar='FLOAT',
    help='Learning rate')
parser.add_argument(
    '--activation', '-a', type=str, default='TANH', metavar='STR',
    help='NonLinearity to use in hidden layers')
parser.add_argument(
    '--gpu', '-g', type=int, default=-1, metavar='INT',
    help='GPU id (-1 for CPU)')
args = parser.parse_args()

train_loader = LoadDataNLI(os.path.join(args.base_dir, args.train % args.parts[0]),
                           os.path.join(args.base_dir, args.train % args.parts[1]),
                           os.path.join(args.base_dir, args.train_labels),
                           dim=args.dim, bsize=args.bsize, shuffle=True, fraction=args.fraction)

dev_loader = LoadDataNLI(os.path.join(args.base_dir, args.dev % args.parts[0]),
                         os.path.join(args.base_dir, args.dev % args.parts[1]),
                         os.path.join(args.base_dir, args.dev_labels),
                         dim=args.dim, bsize=args.bsize, shuffle=False)

# set GPU and random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu < 0:
    print(' - running on cpu')
else:
    print(' - running on gpu {:d}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
print(' - setting seed to {:d}'.format(args.seed))
print(' - lrate is {:f} and bsize {:d}'.format(args.lr, args.bsize))

# create network
net = Net(fname=args.load,
          idim=4*args.dim, odim=3, nhid=args.nhid,
          dropout=args.dropout, gpu=args.gpu,
          activation=args.activation)
if args.gpu >= 0:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=args.lr)

corr_best = 0
# loop multiple times over the dataset
for epoch in range(args.nepoch):

    loss_epoch = 0.0
    print('Ep {:4d}'.format(epoch), end='')
    # for inputs, labels in train_loader:
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.long()
        if args.gpu >= 0:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        net.train(mode=True)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    print(' | loss {:e}'.format(loss_epoch), end='')

    corr, nbex = net.TestCorpus(dev_loader, 'Dev')
    if corr >= corr_best:
        print(' | saved')
        corr_best = corr
        net_best = copy.deepcopy(net)
    else:
        print('')


if 'net_best' in globals():
    if args.save != '':
        torch.save(net_best.cpu(), args.save)
    print('Best Dev: {:d} = {:5.2f}%'
          .format(corr_best, 100.0 * corr_best.float() / nbex))

    if args.gpu >= 0:
        net_best = net_best.cuda()

# test on (several) languages
if args.test is None:
    os.exit()

print('Testing on {}'.format(args.test))
if not args.cross_lingual:
    for l in args.lang:
        test_loader = LoadDataNLI(os.path.join(args.base_dir, args.test % args.parts[0] + '.' + l),
                                  os.path.join(args.base_dir, args.test % args.parts[1] + '.' + l),
                                  os.path.join(args.base_dir, args.test_labels + '.' + l),
                                  dim=args.dim, bsize=args.bsize, shuffle=False, quiet=True)
        print('Ep best | Eval Test lang {:s}'.format(l), end='')
        ofname = args.save_outputs + '.{:s}-{:s}'.format(l, l) + '.txt' if args.save_outputs else None
        net_best.TestCorpus(test_loader, 'Test', out_fname=ofname)
        print('')
else:  # cross-lingual
    err = np.empty((len(args.lang), len(args.lang)), dtype=np.float32)
    i1 = 0
    for l1 in args.lang:
        i2 = 0
        for l2 in args.lang:
            test_loader = LoadDataNLI(os.path.join(args.base_dir, args.test % args.parts[0] + '.' + l1),
                                      os.path.join(args.base_dir, args.test % args.parts[1] + '.' + l2),
                                      os.path.join(args.base_dir, args.test_labels + '.' + l2),
                                      dim=args.dim, bsize=args.bsize, shuffle=False, quiet=True)
            print('Ep best | Eval Test {:s}-{:s}'.format(l1, l2), end='')
            ofname = args.save_outputs + '.{:s}-{:s}'.format(l1, l2) + '.txt' if args.save_outputs else None
            p, n = net_best.TestCorpus(test_loader, 'Test',
                                       out_fname=ofname)
            err[i1, i2] = 100.0 * float(p) / n
            i2 += 1
            print('')
        i1 += 1

    print('\nAccuracy matrix:')
    print('      ', end='')
    for i2 in range(err.shape[1]):
        print('  {:4s} '.format(args.lang[i2]), end='')

    print('  avg')
    for i1 in range(err.shape[0]):
        print('{:4s}'.format(args.lang[i1]), end='')
        for i2 in range(err.shape[1]):
            print('  {:5.2f}'.format(err[i1, i2]), end='')
        print('   {:5.2f}'.format(np.average(err[i1, :])))
    print('avg ', end='')
    for i2 in range(err.shape[1]):
        print('  {:5.2f}'.format(np.average(err[:, i2])), end='')
    print('  {:5.2f}'.format(np.average(err)))

    if err.shape[0] == err.shape[1]:
        s = 0
        # TODO: we assume the first lang is English
        for i1 in range(1, err.shape[0]):
            s += err[i1, i1]
        print('xnli-xx: {:5.2f}'.format(s/(err.shape[0]-1)))

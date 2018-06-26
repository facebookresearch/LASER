#!/usr/bin/python
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Simple MLP classifier for sentence embeddings


import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import numpy as np
import copy

################################################


def LoadData(bdir, dfn, lfn, dim=1024, bsize=32, shuffle=False, quit=False):
    x = np.fromfile(bdir + dfn, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)

    lbl = np.loadtxt(bdir + lfn, dtype=np.int32)
    lbl.reshape(lbl.shape[0], 1)
    lbl -= 1  # convert [1,N] to [0,N-1]
    if not quit:
        print(" - read %dx%d elements in %s" % (x.shape[0], x.shape[1], dfn))
        print(" - read %d labels [%d,%d] in %s"
              % (lbl.shape[0], lbl.min(), lbl.max(), lfn))

    D = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(lbl))
    loader = data_utils.DataLoader(D, batch_size=bsize, shuffle=shuffle)
    return loader

################################################


class Net(nn.Module):
    def __init__(self, idim=1024, odim=2, nhid=0, dropout=0.0, gpu=0):
        super(Net, self).__init__()
        self.gpu = gpu
        modules = []
        if nhid > 0:
            modules.append(nn.Linear(idim, nhid))
            modules.append(nn.Tanh())
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nhid, odim))
            print(" - mlp %d-%d-%d, dropout=%.1f"
                  % (idim, nhid, odim, dropout))
        else:
            modules.append(nn.Linear(idim, odim))
            print(" - mlp %d-%d" % (idim, odim))
        # Softmax is included CrossEntropyLoss !
        self.mlp = nn.Sequential(*modules)

        if self.gpu >= 0:
            self.mlp = self.mlp.cuda()

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dset, name=" Dev", nlbl=4):
        correct = 0
        total = 0
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)
        for data in dset:
            X, Y = data
            Y = Y.long()
            if self.gpu >= 0:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(Variable(X))
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).int().sum()
            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()

        sys.stdout.write(" | %4s: %5.2f%%"
                         % (name, 100.0 * correct.float() / total))
        sys.stdout.write(" | classes:")
        for i in range(nlbl):
            sys.stdout.write(" %5.2f" % (100.0 * corr[i] / total))

        return correct, total


################################################

parser = argparse.ArgumentParser(
           formatter_class=argparse.RawDescriptionHelpFormatter,
           description="Simple sentence classifier")

# Data
parser.add_argument(
    '--base_dir', '-b', type=str, required=True, metavar='PATH',
    help="Directory with all the data files)")
parser.add_argument(
    '--save', '-s', type=str, required=False, metavar='PATH', default="",
    help="File in which to save best network")
parser.add_argument(
    '--train', '-t', type=str, required=True, metavar='STR',
    help="Name of training corpus")
parser.add_argument(
    '--train_labels', '-T', type=str, required=True, metavar='STR',
    help="Name of training corpus (labels)")
parser.add_argument(
    '--dev', '-d', type=str, required=True, metavar='STR',
    help="Name of development corpus")
parser.add_argument(
    '--dev_labels', '-D', type=str, required=True, metavar='STR',
    help="Name of development corpus (labels)")
parser.add_argument(
    '--test', '-e', type=str, required=True, metavar='STR',
    help="Name of test corpus without language extension")
parser.add_argument(
    '--test_labels', '-E', type=str, required=True, metavar='STR',
    help="Name of test corpus without language extension (labels)")
parser.add_argument(
    '--langs', '-l', type=str, required=True, nargs='*', action='append',
    help="List of languages to test on (option can be given multiple times)")

# network definition
parser.add_argument(
    '--nhid', '-n', type=int, default=2, metavar='INT',
    help="Size of hidden layer (0 = no hidden layer)")
parser.add_argument(
    '--dropout', '-o', type=float, default=0.0, metavar='FLOAT',
    help="Value  of dropout")
parser.add_argument(
    '--nepoch', '-N', type=int, default=100, metavar='INT',
    help="Number of epochs")
parser.add_argument(
    '--bsize', '-B', type=int, default=128, metavar='INT',
    help="Batch size")
parser.add_argument(
    '--seed', '-S', type=int, default=123456789, metavar='INT',
    help="Initial random seed")
parser.add_argument(
    '--gpu', '-g', type=int, default=-1, metavar='INT',
    help="GPU id (-1 for CPU)")
args = parser.parse_args()

args.base_dir = args.base_dir + "/"

train_loader = LoadData(args.base_dir, args.train, args.train_labels,
                        bsize=args.bsize, shuffle=True)

dev_loader = LoadData(args.base_dir, args.dev, args.dev_labels,
                      bsize=args.bsize, shuffle=False)

# set GPU and random seed
torch.cuda.set_device(args.gpu)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(" - setting seed to %d" % args.seed)

# create network
net = Net(odim=4, nhid=args.nhid, dropout=args.dropout, gpu=args.gpu)
criterion = nn.CrossEntropyLoss()
if args.gpu >= 0:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), weight_decay=0.0)

corr_best = 0
# loop multiple times over the dataset
for epoch in range(args.nepoch):

    loss_epoch = 0.0
    sys.stdout.write("Ep %4d" % epoch)
    sys.stdout.flush()
    # for inputs, labels in train_loader:
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.long()
        if args.gpu >= 0:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        net.train(mode=True)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.data.item()

    sys.stdout.write(" | loss %e" % loss_epoch)
    sys.stdout.flush()

    corr, nbex = net.TestCorpus(dev_loader, "Dev")
    if corr >= corr_best:
        print(" | saved")
        corr_best = corr
        net_best = copy.deepcopy(net)
    else:
        print("")


if 'net_best' in globals():
    if args.save != "":
        torch.save(net_best.cpu(), args.save)
    print('Best Dev: %d = %5.2f%%'
          % (corr_best, 100.0 * corr_best.float() / nbex))

    if args.gpu >= 0:
        net_best = net_best.cuda()

# test on (several) languages
for l in args.langs:
    test_loader = LoadData(args.base_dir, args.test + "." + l[0],
                           args.test_labels + "." + l[0],
                           bsize=args.bsize, shuffle=False, quit=True)
    sys.stdout.write("Ep best | Eval Test lang %2s" % l[0])
    net_best.TestCorpus(test_loader, "Test")
    print("")

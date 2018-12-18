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
# This module contains an implementation of an BLSTM
# with code to read it in PyTorch format

import sys
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import mlenc_const as const


class BLSTM(nn.Module):

    def __init__(self, fname, gpu=-1,
                 verbose=(const.VERBOSE_INFO & const.VERBOSE_LOAD)):
        super(BLSTM, self).__init__()

        self.verbose = verbose
        self.gpu = gpu
        if fname != "":
            print("Reading network " + fname)
            loaded = torch.load(fname)
            self.nvoc = loaded.nvoc
            self.ninp = loaded.ninp
            self.nhid = loaded.nhid
            self.nlyr = loaded.nlyr
            self.nembed = loaded.nembed
            self.drop = loaded.drop
            self.embed = loaded.embed
            self.blstm = loaded.blstm
            if verbose & const.VERBOSE_LOAD > 0:
                print(" - ninp=%d, nhid=%d, nlyr=%d, dropout=%.2f"
                      % (self.ninp, self.nhid, self.nlyr, self.drop))

            if gpu >= 0:
                print(" - transfer model to GPU %d" % self.gpu)
                torch.cuda.set_device(self.gpu)  # 0.., no-op if negatie
                self.embed.cuda()
                self.blstm.cuda()

        # mod.nvoc      number of entries in lookup table
        # mod.ninp      dimension of word embedding
        # mod.embed     weights of lookup table nvoc x ninp
        #
        # mod.nhid      size of LSTM hidden layer
        # mod.nlyr      number of BLSTM layers
        # mod.nembed    size of output embedding
        # mod.drop      value of dropout
        # mod.blstm     model itself

    #############################
    # encode a whole batch: bsize x seq_len
    def SetParams(self, embed, nlyr, nhid, drop, blstm):

        # word embeddings
        self.embed = embed
        self.nvoc = embed.num_embeddings
        self.ninp = embed.embedding_dim
        self.nembed = 2 * nhid  # for BLSTM only !
        # BLSTM architecture
        self.nlyr = nlyr
        self.nhid = nhid
        self.drop = drop
        self.blstm = blstm

    #############################
    # encode a whole batch: bsize x seq_len
    def EncodeBatch(self, data):
        self.blstm.train(False)
        bsize = data.text_bin.shape[1]
        # worde emdedding expect a Long Tensor !
        if self.gpu < 0:
            binv = Variable(torch.from_numpy(data.text_bin).long(),
                            volatile=False)
        else:
            binv = Variable(torch.from_numpy(data.text_bin).long().cuda(),
                            volatile=False)

        # embed the whole batch
        # binv: seq_len
        #  inp: seq_len x bsize x ninp
        #  out: seq_len x bsize x nembed
        #  enc: bsize x nembed
        inpv = self.embed(binv)

        if self.verbose & const.VERBOSE_DEBUG:
            print("\nNorm of word embeddings:")
            for b in range(bsize):
                sys.stdout.write(" - word embed b%d" % b)
                for s in range(data.text_slen[b]):
                    sys.stdout.write(" %d: %e"
                                     % (data.text_bin[s, b],
                                        inpv.data[s][b].norm()))
                sys.stdout.write("\n")

        # sort by length
        # copy is needed because of negative stride
        idx_sort = np.argsort(data.text_slen)[::-1].copy()
        idx_unsort = np.argsort(idx_sort)
        slen_sorted = np.take(data.text_slen, idx_sort)
        if self.gpu < 0:
            inpv_sorted = inpv.index_select(
                1,
                Variable(torch.from_numpy(idx_sort),
                         volatile=False))
        else:
            inpv_sorted = inpv.index_select(
                1,
                Variable(torch.from_numpy(idx_sort).cuda(),
                         volatile=False))

        inpv_packed = nn.utils.rnn.pack_padded_sequence(inpv_sorted,
                                                        slen_sorted)
        packed_out = self.blstm(inpv_packed)[0]

        # unpack: max_slen x bsize x nembed
        # set unused outputs to minus infinity so they won't infer with maxpool
        unpacked_out = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            padding_value=-np.infty)[0]

        # max and unsort
        text_enc_np = torch.max(unpacked_out.data, dim=0)[0].cpu().numpy()
        data.text_enc = np.take(text_enc_np, idx_unsort, axis=0)

        if self.verbose & const.VERBOSE_DEBUG:
            unpacked_out_unsort = unpacked_out.data.index_select(
               1,
               torch.from_numpy(idx_unsort).cuda())
            for b in range(bsize):
                sys.stdout.write(" - output states b%d" % b)
                for s in range(data.text_slen[b]):
                    sys.stdout.write(" %e" % unpacked_out_unsort[s][b].norm())
                sys.stdout.write("  max=%e\n"
                                 % np.linalg.norm(data.text_enc[b]))

    #############################
    # encode a whole batch: bsize x seq_len
    # Replace batch mode by a simple loop
    def EncodeBatch1(self, data):
        self.blstm.train(False)
        bsize = data.text_bin.shape[1]
        # word emdedding expect a Long Tensor !
        if self.gpu < 0:
            binv = Variable(torch.from_numpy(data.text_bin).long(),
                            volatile=False)
        else:
            binv = Variable(torch.from_numpy(data.text_bin).long().cuda(),
                            volatile=False)

        # embed the whole batch
        # binv: seq_len x bsize
        #  inp: seq_len x bsize x ninp
        #  out: seq_len x bsize x nembed
        #  enc: bsize x nembed
        inpv = self.embed(binv)

        # debugging output
        if self.verbose & const.VERBOSE_DEBUG:
            print("\nNorm of word embeddings:")
            for b in range(bsize):
                sys.stdout.write(" - word embed b%d" % b)
                for s in range(data.text_slen[b]):
                    sys.stdout.write(
                        " %d: %e"
                        % (data.text_bin[-data.text_slen[b]+s, b],
                           inpv.data[-data.text_slen[b]+s][b].norm()))
                sys.stdout.write("\n")

        # embed each sequence INDIVIDUALLY
        # TODO: use batch mode with packed input and output to RNN
        data.text_enc.resize(bsize, self.nembed)
        # print("\nNorm of last states and max over them:") # debugging output
        for b in range(bsize):
            # print("     encode batch %d, len %d" % (b, data.text_slen[b]))
            out, h_n = self.blstm(inpv[-data.text_slen[b]:].narrow(1, b, 1))
            data.text_enc[b] = torch.max(out.data, 0)[0].squeeze().cpu()
            if self.verbose & const.VERBOSE_DEBUG:
                # debugging output
                sys.stdout.write(" - output states b%d" % b)
                for s in range(data.text_slen[b]):
                    sys.stdout.write(" %e" % out.data[s][0].norm())
                sys.stdout.write("  max=%e\n"
                                 % np.linalg.norm(data.text_enc[b]))

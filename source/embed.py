#!/usr/bin/python3
# Copyright (c) 2018-present, Facebook, Inc.
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
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code


import re
import os
import tempfile
import sys
import time
import argparse
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
from text_processing import Token, BPEfastApply


SPACE_NORMALIZER = re.compile("\s+")
Batch = namedtuple('Batch', 'srcs tokens lengths')


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


# TODO Do proper padding from the beginning
def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


class SentenceEncoder:

    def __init__(self, model_path, max_sentences=None, max_tokens=None, cpu=False, fp16=False, sort_kind='quicksort'):
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        self.encoder = Encoder(**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']
        if fp16:
            self.encoder.half()
        if self.use_cuda:
            self.encoder.cuda()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
        self.encoder.eval()
        embeddings = self.encoder(tokens, lengths)['sentemb']
        return embeddings.detach().cpu().numpy()

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        ids = torch.LongTensor(ntokens + 1)
        for i, token in enumerate(tokens):
            ids[i] = self.dictionary.get(token, self.unk_index)
        ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            for i in range(len(tokens)):
                toks[i, -tokens[i].shape[0]:] = tokens[i]
            return Batch(
                srcs=None,
                tokens=toks,
                lengths=torch.LongTensor(lengths)
            ), indices

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and ((self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens) or
                                   (self.max_sentences is not None and nsentences == self.max_sentences)):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class Encoder(nn.Module):
    def __init__(
        self, num_embeddings, padding_idx, embed_dim=320, hidden_size=512, num_layers=1, bidirectional=False,
        left_pad=True, padding_value=0.
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ], dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            'sentemb': sentemb,
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }


def EncodeLoad(args):
    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(' - loading encoder', args.encoder)
    return SentenceEncoder(args.encoder,
               max_sentences=args.max_sentences,
               max_tokens=args.max_tokens,
               cpu=args.cpu)


def EncodeTime(t):
    t = int(time.time() - t) 
    if t < 1000:
        print(' in {:d}s'.format(t))
    else:
        print(' in {:d}m{:d}s'.format(t // 60, t % 60))

# Encode sentences (existing file pointers)
def EncodeFilep(encoder, inp_file, out_file, buffer_size=10000, verbose=False):
    n = 0
    t = time.time()
    for sentences in buffered_read(inp_file, buffer_size):
        encoder.encode_sentences(sentences).tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            print('\r - Encoder: {:d} sentences'.format(n), end='')
    if verbose:
        print('\r - Encoder: {:d} sentences'.format(n), end='')
        EncodeTime(t)

# Encode sentences (file names)
def EncodeFile(encoder, inp_fname, out_fname,
              buffer_size=10000, verbose=False, over_write=False,
              inp_encoding='utf-8'):
    # TODO :handle over write
    if not os.path.isfile(out_fname):
        if verbose:
            print(' - Encoder: {} to {}'.
                  format(os.path.basename(inp_fname) if len(inp_fname) > 0 else 'stdin',
                         os.path.basename(out_fname)))
        fin = open(inp_fname, 'r', encoding=inp_encoding, errors='surrogateescape') if len(inp_fname) > 0 else sys.stdin
        fout = open(out_fname, mode='wb')
        EncodeFilep(encoder, fin, fout, buffer_size=buffer_size, verbose=verbose)
        fin.close()
        fout.close()
    elif not over_write and verbose:
        print(' - Encoder: {} exists already'.format(os.path.basename(out_fname)))

# Load existing embeddings
def EmbedLoad(fname, dim=1024, verbose=False):
    x = np.fromfile(fname, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(' - Embeddings: {:s}, {:d}x{:d}'.format(fname, x.shape[0], dim))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LASER: Embed sentences')
    parser.add_argument('--encoder', type=str, required=True,
        help='encoder to be used')
    parser.add_argument('--token-lang', type=str, default='--',
        help="Perform tokenization with given language ('--' for no tokenization)")
    parser.add_argument('--bpe-codes', type=str, default=None,
        help='Apply BPE using specified codes')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Detailed output')

    parser.add_argument('-o', '--output', required=True,
        help='Output sentence embeddings')
    parser.add_argument('--buffer-size', type=int, default=10000,
        help='Buffer size (sentences)')
    parser.add_argument('--max-tokens', type=int, default=12000,
        help='Maximum number of tokens to process in a batch')
    parser.add_argument('--max-sentences', type=int, default=None,
        help='Maximum number of sentences to process in a batch')
    parser.add_argument('--cpu', action='store_true',
        help='Use CPU instead of GPU')
    parser.add_argument('--stable', action='store_true',
        help='Use stable merge sort instead of quick sort')
    args = parser.parse_args()

    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    if args.verbose:
        print(' - Encoder: loading {}'.format(args.encoder))
    encoder = SentenceEncoder(args.encoder,
                              max_sentences=args.max_sentences,
                              max_tokens=args.max_tokens,
                              sort_kind='mergesort' if args.stable else 'quicksort',
                              cpu=args.cpu)

    with tempfile.TemporaryDirectory() as tmpdir:
        ifname = ''  # stdin will be used
        if args.token_lang != '--':
            tok_fname = os.path.join(tmpdir, 'tok')
            Token(ifname,
                  tok_fname,
                  lang=args.token_lang,
                  romanize=True if args.token_lang == 'el' else False,
                  lower_case=True, gzip=False,
                  verbose=args.verbose, over_write=False)
            ifname = tok_fname

        if args.bpe_codes:
            bpe_fname = os.path.join(tmpdir, 'bpe')
            BPEfastApply(ifname,
                         bpe_fname,
                         args.bpe_codes,
                         verbose=args.verbose, over_write=False)
            ifname = bpe_fname

        EncodeFile(encoder,
                   ifname,
                   args.output,
                   verbose=args.verbose, over_write=False,
                   buffer_size=args.buffer_size) 

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
# Helper functions for tokenization and BPE

import os
import sys
from pathlib import Path
import fastBPE
import numpy as np
from subprocess import run, check_output, DEVNULL

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

FASTBPE = LASER + '/tools-external/fastBPE/fast'
MOSES_BDIR = LASER + '/tools-external/moses-tokenizer/tokenizer/'
MOSES_TOKENIZER = MOSES_BDIR + 'tokenizer.perl -q -no-escape -threads 20 -l '
NORM_PUNC = MOSES_BDIR + 'normalize-punctuation.perl -l '
DESCAPE = MOSES_BDIR + 'deescape-special-chars.perl'
REM_NON_PRINT_CHAR = MOSES_BDIR + 'remove-non-printing-char.perl'
SPM_DIR = LASER + '/tools-external/sentencepiece-master/build/src/'
SPM = 'LD_LIBRARY_PATH=' + SPM_DIR + ' ' + SPM_DIR + '/spm_encode --output_format=piece'

# Romanization (Greek only)
ROMAN_LC = 'python3 ' + LASER + '/source/lib/romanize_lc.py -l '

# Mecab tokenizer for Japanese
MECAB = LASER + '/tools-external/mecab'




###############################################################################
#
# Tokenize a line of text
#
###############################################################################

def TokenLine(line, lang='en', lower_case=True, romanize=False):
    assert lower_case, 'lower case is needed by all the models'
    roman = lang if romanize else 'none'
    tok = check_output(
            REM_NON_PRINT_CHAR
            + '|' + NORM_PUNC + lang
            + '|' + DESCAPE
            + '|' + MOSES_TOKENIZER + lang
            + ('| python3 -m jieba -d ' if lang == 'zh' else '')
            + ('|' + MECAB + '/bin/mecab -O wakati -b 50000 ' if lang == 'ja' else '')
            + '|' + ROMAN_LC + roman,
            input=line,
            encoding='UTF-8',
            shell=True)
    return tok.strip()


###############################################################################
#
# Tokenize a file
#
###############################################################################

def Token(inp_fname, out_fname, lang='en',
          lower_case=True, romanize=False, descape=False,
          verbose=False, over_write=False, gzip=False):
    assert lower_case, 'lower case is needed by all the models'
    assert not over_write, 'over-write is not yet implemented'
    if not os.path.isfile(out_fname):
        cat = 'zcat ' if gzip else 'cat '
        roman = lang if romanize else 'none'
        # handle some iso3 langauge codes
        if lang in ('cmn', 'wuu', 'yue'):
            lang = 'zh'
        if lang in ('jpn'):
            lang = 'ja'
        if verbose:
            print(' - Tokenizer: {} in language {} {} {}'
                  .format(os.path.basename(inp_fname), lang,
                          '(gzip)' if gzip else '',
                          '(de-escaped)' if descape else '',
                          '(romanized)' if romanize else ''))
        run(cat + inp_fname
            + '|' + REM_NON_PRINT_CHAR
            + '|' + NORM_PUNC + lang
            + ('|' + DESCAPE if descape else '')
            + '|' + MOSES_TOKENIZER + lang
            + ('| python3 -m jieba -d ' if lang == 'zh' else '')
            + ('|' + MECAB + '/bin/mecab -O wakati -b 50000 ' if lang == 'ja' else '')
            + '|' + ROMAN_LC + roman
            + '>' + out_fname,
            env=dict(os.environ, LD_LIBRARY_PATH=MECAB + '/lib'),
            shell=True)
    elif not over_write and verbose:
        print(' - Tokenizer: {} exists already'
              .format(os.path.basename(out_fname), lang))


###############################################################################
#
# Apply SPM on a whole file
#
###############################################################################

def SPMApply(inp_fname, out_fname, spm_model, lang='en',
             lower_case=True, descape=False,
             verbose=False, over_write=False, gzip=False):
    assert lower_case, 'lower case is needed by all the models'
    if not os.path.isfile(out_fname):
        cat = 'zcat ' if gzip else 'cat '
        if verbose:
            print(' - SPM: processing {} {} {}'
                  .format(os.path.basename(inp_fname),
                         '(gzip)' if gzip else '',
                         '(de-escaped)' if descape else ''))

        if not os.path.isfile(spm_model):
            print(' - SPM: model {} not found'.format(spm_model))
        check_output(cat + inp_fname
            + '|' + REM_NON_PRINT_CHAR
            + '|' + NORM_PUNC + lang
            + ('|' + DESCAPE if descape else '')
            + '|' + ROMAN_LC + 'none'
            + '|' + SPM + " --model=" + spm_model
            + ' > ' + out_fname,
            shell=True, stderr=DEVNULL)
    elif not over_write and verbose:
        print(' - SPM: {} exists already'
              .format(os.path.basename(out_fname)))


###############################################################################
#
# Apply FastBPE on one line of text
#
###############################################################################

def BPEfastLoad(bpe_codes):
    bpe_vocab = bpe_codes.replace('fcodes', 'fvocab')
    return fastBPE.fastBPE(bpe_codes, bpe_vocab)

def BPEfastApplyLine(line, bpe):
    return bpe.apply([line])[0]


###############################################################################
#
# Apply FastBPE on a whole file
#
###############################################################################

def BPEfastApply(inp_fname, out_fname, bpe_codes,
                 verbose=False, over_write=False):
    if not os.path.isfile(out_fname):
        if verbose:
            print(' - fast BPE: processing {}'
                  .format(os.path.basename(inp_fname)))
        bpe_vocab = bpe_codes.replace('fcodes', 'fvocab')
        if not os.path.isfile(bpe_vocab):
            print(' - fast BPE: focab file not found {}'.format(bpe_vocab))
            bpe_vocab = ''
        run(FASTBPE + ' applybpe '
            + out_fname + ' ' + inp_fname
            + ' ' + bpe_codes
            + ' ' + bpe_vocab, shell=True, stderr=DEVNULL)
    elif not over_write and verbose:
        print(' - fast BPE: {} exists already'
              .format(os.path.basename(out_fname)))


###############################################################################
#
# Split long lines into multiple sentences at "."
#
###############################################################################

def SplitLines(ifname, of_txt, of_sid):
    if os.path.isfile(of_txt):
        print(' - SplitLines: {} already exists'.format(of_txt))
        return
    nl = 0
    nl_sp = 0
    maxw = 0
    maxw_sp = 0
    fp_sid = open(of_sid, 'w')
    fp_txt = open(of_txt, 'w')
    with open(ifname, 'r') as ifp:
        for line in ifp:
            print('{:d}'.format(nl), file=fp_sid)  # store current sentence ID
            nw = 0
            words = line.strip().split()
            maxw = max(maxw, len(words))
            for i, word in enumerate(words):
                if word == '.' and i != len(words)-1:
                    if nw > 0:
                        print(' {}'.format(word), file=fp_txt)
                    else:
                        print('{}'.format(word), file=fp_txt)
                    # store current sentence ID
                    print('{:d}'.format(nl), file=fp_sid)
                    nl_sp += 1
                    maxw_sp = max(maxw_sp, nw+1)
                    nw = 0
                else:
                    if nw > 0:
                        print(' {}'.format(word), end='', file=fp_txt)
                    else:
                        print('{}'.format(word), end='', file=fp_txt)
                    nw += 1
            if nw > 0:
                # handle remainder of sentence
                print('', file=fp_txt)
                nl_sp += 1
                maxw_sp = max(maxw_sp, nw+1)
            nl += 1
    print(' - Split sentences: {}'.format(ifname))
    print(' -                  lines/max words: {:d}/{:d} -> {:d}/{:d}'
          .format(nl, maxw, nl_sp, maxw_sp))
    fp_sid.close()
    fp_txt.close()


###############################################################################
#
# Join embeddings of previously split lines (average)
#
###############################################################################

def JoinEmbed(if_embed, sid_fname, of_embed, dim=1024):
    if os.path.isfile(of_embed):
        print(' - JoinEmbed: {} already exists'.format(of_embed))
        return
    # read the input embeddings
    em_in = np.fromfile(if_embed, dtype=np.float32, count=-1).reshape(-1, dim)
    ninp = em_in.shape[0]
    print(' - Combine embeddings:')
    print('                input: {:s} {:d} sentences'.format(if_embed, ninp))

    # get all sentence IDs
    sid = np.empty(ninp, dtype=np.int32)
    i = 0
    with open(sid_fname, 'r') as fp_sid:
        for line in fp_sid:
            sid[i] = int(line)
            i += 1
    nout = sid.max() + 1
    print('                IDs: {:s}, {:d} sentences'.format(sid_fname, nout))

    # combining
    em_out = np.zeros((nout, dim), dtype=np.float32)
    cnt = np.zeros(nout, dtype=np.int32)
    for i in range(ninp):
        idx = sid[i]
        em_out[idx] += em_in[i]  # cumulate sentence vectors
        cnt[idx] += 1

    if (cnt == 0).astype(int).sum() > 0:
        print('ERROR: missing lines')
        sys.exit(1)

    # normalize
    for i in range(nout):
        em_out[i] /= cnt[i]

    print('                output: {:s}'.format(of_embed))
    em_out.tofile(of_embed)

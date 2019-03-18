#!/bin/bash
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
# Python tools for BUCC bitext mining

import argparse

###############################################################################
#
# Find te optimal threshold given gold alignments
#
###############################################################################

def BuccOptimize(candidate2score, gold):
    items = sorted(candidate2score.items(), key=lambda x: -x[1])
    ngold = len(gold)
    nextract = ncorrect = 0
    threshold = 0
    best_f1 = 0
    for i in range(len(items)):
        nextract += 1
        if '\t'.join(items[i][0]) in gold:
            ncorrect += 1
        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / ngold
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                threshold = (items[i][1] + items[i + 1][1]) / 2
    return threshold


###############################################################################
#
# Main
#
###############################################################################

parser = argparse.ArgumentParser(description='LASER: tools for BUCC bitext mining')
parser.add_argument('--encoding', default='utf-8',
    help='character encoding for input/output')
parser.add_argument('--src-lang', required=True,
    help='the source language id')
parser.add_argument('--trg-lang', required=True,
    help='the target language id')
parser.add_argument('--bucc-texts', required=True,
    help='Base name of the text files (language added)')
parser.add_argument('--bucc-ids', required=True,
    help='Base name of the ID files (language added)')
parser.add_argument('--candidates', required=True,
    help='File name of candidate alignments')
parser.add_argument('--gold', default=None,
    help='File name of gold alignments')
parser.add_argument('--threshold', type=float, default=-1,
    help='Threshold (used with --output)')
parser.add_argument('--output', default=None,
    help='File name of output alignments which are below threshold')
parser.add_argument('--verbose', action='store_true',
    help='Detailed output')
args = parser.parse_args()

print('LASER: tools for BUCC bitext mining')

assert (args.gold or args.threshold > 0) \
       and not (args.gold and args.threshold > 0), \
       'Either "--gold" or "--threshold" must be specified'
if args.verbose:
    print(' - reading sentences and IDs')

src_sent2id, trg_sent2id = {}, {}
for lang, sent2id in (args.src_lang, src_sent2id), (args.trg_lang, trg_sent2id):
    repeated = set()
    with open(args.bucc_texts + '.' + lang, encoding=args.encoding, errors='surrogateescape') as f:
        sentences = [line.strip() for line in f]
    with open(args.bucc_ids + '.' + lang, encoding=args.encoding, errors='surrogateescape') as f:
        ids = [line.strip() for line in f]
    for id, sent in zip(ids, sentences):
        if sent in sent2id:
            repeated.add(sent)
        else:
            sent2id[sent] = id
    for sent in repeated:
        del sent2id[sent]

if args.verbose:
    print(' - reading candidates {}'.format(args.candidates))
candidate2score = {}
# id2txt = {}
with open(args.candidates, encoding=args.encoding, errors='surrogateescape') as f:
    for line in f:
        score, src, trg = line.split('\t')
        score = float(score)
        src = src.strip()
        trg = trg.strip()
        if src in src_sent2id and trg in trg_sent2id:
            src_id = src_sent2id[src]
            trg_id = trg_sent2id[trg]
            score = max(score, candidate2score.get((src_id, trg_id), score))
            candidate2score[(src_id, trg_id)] = score
            # id2txt[src_id + '\t' + trg_id] = src + '\t' + trg

def BuccExtract(cand2score, th, fname):
    if fname:
        of = open(fname, 'w', encoding=args.encoding)
    bitexts = []
    for (src, trg), score in cand2score.items():
        if score >= th:
            bitexts.append(src + '\t' + trg)
            if fname:
                of.write(src + '\t' + trg + '\n')
    if fname:
        of.close()
    return bitexts

if args.gold:
    if args.verbose:
        print(' - optimizing threshold on gold alignments {}'.format(args.gold))
        if args.output:
            print(' - extracted bitext are written into {:s}'.format(args.output))
    gold = {line.strip() for line in open(args.gold)}
    threshold = BuccOptimize(candidate2score, gold)

    bitexts = BuccExtract(candidate2score, threshold, args.output)
    ncorrect = len(gold.intersection(bitexts))
    if ncorrect > 0:
        precision = ncorrect / len(bitexts)
        recall = ncorrect / len(gold)
        f1 = 2*precision*recall / (precision + recall)
    else:
        precision = recall = f1 = 0

    print(' - best threshold={:f}: precision={:.2f}, recall={:.2f}, F1={:.2f}'
          .format(threshold, 100*precision, 100*recall, 100*f1))


if args.threshold > 0:
    if args.verbose:
        print(' - extracting bitexts for threshold {:f} into {:s}'.format(args.threshold, args.output))
    BuccExtract(candidate2score, args.threshold, args.output)

#!/usr/bin/python3
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
# Tool to calculate multilingual similarity error rate
# on various predefined test sets


import os
import argparse
import pandas
import tempfile
import numpy as np
import itertools
import logging
import sys
from typing import List, Tuple
from tabulate import tabulate
from pathlib import Path
from xsim import xSIM
from embed import embed_sentences, SentenceEncoder, HuggingFaceEncoder, load_model

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger('eval')

class Eval:
    def __init__(self, args):
        self.base_dir = args.base_dir
        self.corpus = args.corpus
        self.split = args.corpus_part
        self.min_sents = args.min_sents
        self.index_comparison = args.index_comparison
        self.emb_dimension = args.embedding_dimension
        self.encoder_args = {
            k: v
            for k, v in args._get_kwargs()
            if k
            in ["max_sentences", "max_tokens", "cpu", "fp16", "sort_kind", "verbose"]
        }
        self.src_bpe_codes = args.src_bpe_codes
        self.tgt_bpe_codes = args.tgt_bpe_codes
        self.src_spm_model = args.src_spm_model
        self.tgt_spm_model = args.tgt_spm_model
        logger.info('loading src encoder')
        self.src_encoder = load_model(
            args.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
            hugging_face=args.use_hugging_face,
            **self.encoder_args,
        )
        if args.tgt_encoder:
            logger.info('loading tgt encoder')
            self.tgt_encoder = load_model(
                args.tgt_encoder,
                self.tgt_spm_model,
                self.tgt_bpe_codes,
                hugging_face=args.use_hugging_face,
                **self.encoder_args,
            )
        else:
            logger.info('encoding tgt using src encoder')
            self.tgt_encoder = self.src_encoder
            self.tgt_bpe_codes = self.src_bpe_codes
            self.tgt_spm_model = self.src_spm_model
        self.nway = args.nway
        self.buffer_size = args.buffer_size
        self.fp16 = args.fp16
        self.margin = args.margin

    def _embed(self, tmpdir, langs, encoder, spm_model, bpe_codes) -> List[List[str]]:
        emb_data = []
        for lang in langs:
            fname = f"{lang}.{self.split}"
            infile = os.path.join(self.base_dir, self.corpus, self.split, fname)
            outfile = os.path.join(tmpdir, fname)
            embed_sentences(
                infile,
                outfile,
                encoder=encoder,
                spm_model=spm_model,
                bpe_codes=bpe_codes,
                token_lang=lang if bpe_codes else "--",
                buffer_size=self.buffer_size,
                **self.encoder_args,
            )
            assert (
                os.path.isfile(outfile) and os.path.getsize(outfile) > 0
            ), f"Error encoding {infile}"
            emb_data.append([lang, infile, outfile])
        return emb_data

    def _xsim(self, src_emb, src_lang, tgt_emb, tgt_lang, tgt_txt) -> Tuple[int, int]:
        err, nbex = xSIM(
            src_emb,
            tgt_emb,
            margin=self.margin,
            dim=self.emb_dimension,
            fp16=self.fp16,
            eval_text=tgt_txt if not self.index_comparison else None,
        )
        return err, nbex

    def calc_xsim(self, embdir, src_langs, tgt_langs, err_sum=0, totl_nbex=0) -> None:
        outputs = []
        src_emb_data = self._embed(
            embdir,
            src_langs,
            self.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
        )
        tgt_emb_data = self._embed(
            embdir,
            tgt_langs,
            self.tgt_encoder,
            self.tgt_spm_model,
            self.tgt_bpe_codes,
        )
        combs = list(itertools.product(src_emb_data, tgt_emb_data))
        for (src_lang, src_txt, src_emb), (tgt_lang, tgt_txt, tgt_emb) in combs:
            if src_lang == tgt_lang:
                continue
            err, nbex = self._xsim(src_emb, src_lang, tgt_emb, tgt_lang, tgt_txt)
            result = round(100 * err / nbex, 2)
            if nbex < self.min_sents:
                result = "skipped"
            else:
                err_sum += err
                totl_nbex += nbex
            outputs.append(
                [self.corpus, f"{src_lang}-{tgt_lang}", f"{result}", f"{nbex}"]
            )
        outputs.append(
            [
                self.corpus,
                "average",
                f"{round(100 * err_sum / totl_nbex, 2)}",
                f"{len(combs)}",
            ]
        )
        print(
            tabulate(
                outputs, tablefmt="psql", headers=["dataset", "src-tgt", "xsim", "nbex"]
            )
        )

    def calc_xsim_nway(self, embdir, langs) -> None:
        err_matrix = np.zeros((len(langs), len(langs)))
        emb_data = self._embed(
            embdir,
            langs,
            self.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
        )
        for i1, (src_lang, src_txt, src_emb) in enumerate(emb_data):
            for i2, (tgt_lang, tgt_txt, tgt_emb) in enumerate(emb_data):
                if src_lang == tgt_lang:
                    err_matrix[i1, i2] = 0
                else:
                    err, nbex = self._xsim(src_emb, src_lang, tgt_emb, tgt_lang, tgt_txt)
                    err_matrix[i1, i2] = 100 * err / nbex
        df = pandas.DataFrame(err_matrix, columns=langs, index=langs)
        df.loc["avg"] = df.sum() / float(df.shape[0] - 1)  # exclude diagonal in average
        print(f"\n{tabulate(df, langs, floatfmt='.2f', tablefmt='grid')}\n\n")
        print(f"Global average: {df.loc['avg'].mean():.2f}")


def run_eval(args) -> None:
    evaluation = Eval(args)
    tmp_dir = None
    if args.embed_dir:
        os.makedirs(args.embed_dir, exist_ok=True)
        embed_dir = args.embed_dir
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        embed_dir = tmp_dir.name
    src_langs = sorted(args.src_langs.split(","))
    if evaluation.nway:
        evaluation.calc_xsim_nway(embed_dir, src_langs)
    else:
        assert (
            args.tgt_langs
        ), "Please provide tgt langs when not performing n-way comparison"
        tgt_langs = sorted(args.tgt_langs.split(","))
        evaluation.calc_xsim(embed_dir, src_langs, tgt_langs)
    if tmp_dir:
        tmp_dir.cleanup()  # remove temporary directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LASER: multilingual similarity error evaluation"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for evaluation files",
        required=True,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Name of evaluation corpus",
        required=True,
    )
    parser.add_argument(
        "--corpus-part",
        type=str,
        default=None,
        help="Specify split of the corpus to use e.g., dev",
        required=True,
    )
    parser.add_argument(
        "--margin",
        type=str,
        default=None,
        help="Margin for xSIM calculation. See: https://aclanthology.org/P19-1309",
    )
    parser.add_argument(
        "--min-sents",
        type=int,
        default=100,
        help="Only use test sets which have at least N sentences",
    )
    parser.add_argument(
        "--nway", action="store_true", help="Test N-way for corpora which support it"
    )
    parser.add_argument(
        "--embed-dir",
        type=str,
        default=None,
        help="Store/load embeddings from specified directory (default temporary)",
    )
    parser.add_argument(
        "--index-comparison",
        action="store_true",
        help="Use index comparison instead of texts (not recommended when test data contains duplicates)",
    )
    parser.add_argument("--src-spm-model", type=str, default=None)
    parser.add_argument("--tgt-spm-model", type=str, default=None)
    parser.add_argument(
        "--src-bpe-codes",
        type=str,
        default=None,
        help="Path to bpe codes for src model",
    )
    parser.add_argument(
        "--tgt-bpe-codes",
        type=str,
        default=None,
        help="Path to bpe codes for tgt model",
    )
    parser.add_argument("--src-encoder", type=str, default=None, required=True)
    parser.add_argument("--tgt-encoder", type=str, default=None)
    parser.add_argument(
        "--buffer-size", type=int, default=100, help="Buffer size (sentences)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=12000,
        help="Maximum number of tokens to process in a batch",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences to process in a batch",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")

    parser.add_argument(
        "--src-langs",
        type=str,
        default=None,
        help="Source-side languages for evaluation",
        required=True,
    )
    parser.add_argument(
        "--tgt-langs",
        type=str,
        default=None,
        help="Target-side languages for evaluation",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store embedding matrices in fp16 instead of fp32",
    )
    parser.add_argument(
        "--sort-kind",
        type=str,
        default="quicksort",
        choices=["quicksort", "mergesort"],
        help="Algorithm used to sort batch by length",
    )
    parser.add_argument(
        "--use-hugging-face",
        action="store_true",
        help="Use a HuggingFace sentence transformer",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=1024,
        help="Embedding dimension for encoders",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")
    args = parser.parse_args()
    run_eval(args)

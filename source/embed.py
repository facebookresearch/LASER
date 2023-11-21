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
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code


import argparse
import logging
import os
import re
import sys
import tempfile
import time
from collections import namedtuple
from pathlib import Path
from subprocess import run
from typing import Optional, Union

assert os.environ.get("LASER"), "Please set the environment variable LASER"
LASER = os.environ["LASER"]
sys.path.append(LASER)

import numpy as np
from lib.text_processing import BPEfastApply, SPMApply, Token
from laser_encoders.models import SentenceEncoder

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("embed")


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


class HuggingFaceEncoder:
    def __init__(self, encoder_name: str, verbose=False):
        from sentence_transformers import SentenceTransformer

        encoder = f"sentence-transformers/{encoder_name}"
        if verbose:
            logger.info(f"loading HuggingFace encoder: {encoder}")
        self.encoder = SentenceTransformer(encoder)

    def encode_sentences(self, sentences):
        return self.encoder.encode(sentences)


def load_model(
    encoder: str,
    spm_model: str,
    bpe_codes: str,
    hugging_face=False,
    verbose=False,
    **encoder_kwargs,
) -> Union[SentenceEncoder, HuggingFaceEncoder]:
    if hugging_face:
        return HuggingFaceEncoder(encoder, verbose=verbose)
    if spm_model:
        spm_vocab = str(Path(spm_model).with_suffix(".cvocab"))
        if verbose:
            logger.info(f"spm_model: {spm_model}")
            logger.info(f"spm_cvocab: {spm_vocab}")
    else:
        spm_vocab = None
    return SentenceEncoder(
        encoder, spm_vocab=spm_vocab, verbose=verbose, **encoder_kwargs
    )


def EncodeLoad(args):
    args.buffer_size = max(args.buffer_size, 1)
    assert (
        not args.max_sentences or args.max_sentences <= args.buffer_size
    ), "--max-sentences/--batch-size cannot be larger than --buffer-size"

    print(" - loading encoder", args.encoder)
    return SentenceEncoder(
        args.encoder,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
        cpu=args.cpu,
        verbose=args.verbose,
    )


def EncodeTime(t):
    t = int(time.time() - t)
    if t < 1000:
        return "{:d}s".format(t)
    else:
        return "{:d}m{:d}s".format(t // 60, t % 60)


# Encode sentences (existing file pointers)
def EncodeFilep(
    encoder, inp_file, out_file, buffer_size=10000, fp16=False, verbose=False
):
    n = 0
    t = time.time()
    for sentences in buffered_read(inp_file, buffer_size):
        encoded = encoder.encode_sentences(sentences)
        if fp16:
            encoded = encoded.astype(np.float16)
        encoded.tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            logger.info("encoded {:d} sentences".format(n))
    if verbose:
        logger.info(f"encoded {n} sentences in {EncodeTime(t)}")


# Encode sentences (file names)
def EncodeFile(
    encoder,
    inp_fname,
    out_fname,
    buffer_size=10000,
    fp16=False,
    verbose=False,
    over_write=False,
    inp_encoding="utf-8",
):
    # TODO :handle over write
    if not os.path.isfile(out_fname):
        if verbose:
            logger.info(
                "encoding {} to {}".format(
                    inp_fname if len(inp_fname) > 0 else "stdin",
                    out_fname,
                )
            )
        fin = (
            open(inp_fname, "r", encoding=inp_encoding, errors="surrogateescape")
            if len(inp_fname) > 0
            else sys.stdin
        )
        fout = open(out_fname, mode="wb")
        EncodeFilep(
            encoder, fin, fout, buffer_size=buffer_size, fp16=fp16, verbose=verbose
        )
        fin.close()
        fout.close()
    elif not over_write and verbose:
        logger.info("encoder: {} exists already".format(os.path.basename(out_fname)))


# Load existing embeddings
def EmbedLoad(fname, dim=1024, verbose=False, fp16=False):
    x = np.fromfile(fname, dtype=(np.float16 if fp16 else np.float32), count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(" - Embeddings: {:s}, {:d}x{:d}".format(fname, x.shape[0], dim))
    return x


# Get memory mapped embeddings
def EmbedMmap(fname, dim=1024, dtype=np.float32, verbose=False):
    nbex = int(os.path.getsize(fname) / dim / np.dtype(dtype).itemsize)
    E = np.memmap(fname, mode="r", dtype=dtype, shape=(nbex, dim))
    if verbose:
        print(" - embeddings on disk: {:s} {:d} x {:d}".format(fname, nbex, dim))
    return E


def embed_sentences(
    ifname: str,
    output: str,
    encoder: Union[SentenceEncoder, HuggingFaceEncoder] = None,
    encoder_path: str = None,
    hugging_face=False,
    token_lang: Optional[str] = "--",
    bpe_codes: Optional[str] = None,
    spm_lang: Optional[str] = "en",
    spm_model: Optional[str] = None,
    verbose: bool = False,
    buffer_size: int = 10000,
    max_tokens: int = 12000,
    max_sentences: Optional[int] = None,
    cpu: bool = False,
    fp16: bool = False,
    sort_kind: str = "quicksort",
):
    assert encoder or encoder_path, "Provide initialised encoder or encoder_path"
    buffer_size = max(buffer_size, 1)
    assert (
        not max_sentences or max_sentences <= buffer_size
    ), "--max-sentences/--batch-size cannot be larger than --buffer-size"

    assert not (bpe_codes and spm_model), "Cannot specify both spm and bpe"

    if encoder_path:
        encoder = load_model(
            encoder_path,
            spm_model,
            bpe_codes,
            verbose=verbose,
            hugging_face=hugging_face,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind=sort_kind,
            cpu=cpu,
        )
    if not ifname:
        ifname = ""  # default to stdin
    with tempfile.TemporaryDirectory() as tmpdir:
        if token_lang != "--":
            tok_fname = os.path.join(tmpdir, "tok")
            Token(
                ifname,
                tok_fname,
                lang=token_lang,
                romanize=True if token_lang == "el" else False,
                lower_case=True,
                gzip=False,
                verbose=verbose,
                over_write=False,
            )
            ifname = tok_fname

        if bpe_codes:
            if ifname == "":  # stdin
                ifname = os.path.join(tmpdir, "no_tok")
                run(f"cat > {ifname}", shell=True)
            bpe_fname = os.path.join(tmpdir, "bpe")
            BPEfastApply(
                ifname, bpe_fname, bpe_codes, verbose=verbose, over_write=False
            )
            ifname = bpe_fname

        if spm_model:
            spm_fname = os.path.join(tmpdir, "spm")
            SPMApply(
                ifname,
                spm_fname,
                spm_model,
                lang=spm_lang,
                lower_case=True,
                verbose=verbose,
                over_write=False,
            )
            ifname = spm_fname

        EncodeFile(
            encoder,
            ifname,
            output,
            verbose=verbose,
            over_write=False,
            buffer_size=buffer_size,
            fp16=fp16,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Embed sentences")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input text file",
    )
    parser.add_argument("--encoder", type=str, required=True, help="encoder to be used")
    parser.add_argument(
        "--token-lang",
        type=str,
        default="--",
        help="Perform tokenization with given language ('--' for no tokenization)",
    )
    parser.add_argument(
        "--bpe-codes", type=str, default=None, help="Apply BPE using specified codes"
    )
    parser.add_argument(
        "--spm-lang", type=str, default="en", help="Apply SPM using specified language"
    )
    parser.add_argument(
        "--spm-model", type=str, default=None, help="Apply SPM using specified model"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")

    parser.add_argument(
        "-o", "--output", required=True, help="Output sentence embeddings"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Buffer size (sentences)"
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
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store embedding matrices in fp16 instead of fp32",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
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

    args = parser.parse_args()
    embed_sentences(
        ifname=args.input,
        encoder_path=args.encoder,
        token_lang=args.token_lang,
        bpe_codes=args.bpe_codes,
        spm_lang=args.spm_lang,
        hugging_face=args.use_hugging_face,
        spm_model=args.spm_model,
        verbose=args.verbose,
        output=args.output,
        buffer_size=args.buffer_size,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        cpu=args.cpu,
        fp16=args.fp16,
        sort_kind=args.sort_kind,
    )

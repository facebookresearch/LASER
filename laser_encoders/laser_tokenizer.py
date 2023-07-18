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
# Helper functions for tokenization

import sys
import gzip
import logging
from pathlib import Path
import typing as tp
import sentencepiece as spm
from sacremoses import MosesPunctNormalizer, MosesDetokenizer

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("preprocess")


class LaserTokenizer:
    def __init__(
        self,
        spm_model: Path,
        lang: str = "en",
        lower_case: bool = True,
        descape: bool = False,
        verbose: bool = False,
        over_write: bool = False,
    ):
        self.spm_model = spm_model
        self.lang = lang
        self.lower_case = lower_case
        self.descape = descape
        self.verbose = verbose
        self.over_write = over_write

        assert spm_model.exists(), f"spm model file: {spm_model} does not exist"
        self.moses_punct_normalizer = MosesPunctNormalizer(self.lang)
        self.moses_detokenizer = MosesDetokenizer()
        self.spm_encoder = spm.SentencePieceProcessor(model_file=str(self.spm_model))

    def open(self, file: Path, mode: str, encoding="utf-8") -> tp.IO:
        return (
            gzip.open(file, mode, encoding=encoding)
            if file.name.endswith(".gz")
            else open(file, mode, encoding=encoding)
        )

    def log(self, message):
        if self.verbose:
            logger.info(message)

    def tokenize(self, text: str) -> str:
        # Preprocessing
        sentence_text = "".join(c for c in text if c.isprintable)
        sentence_text = self.moses_punct_normalizer.normalize(sentence_text)
        if self.descape:
            sentence_text = self.moses_detokenizer.unescape_xml(text=sentence_text)
        if self.lower_case:
            sentence_text = sentence_text.lower()

        # SentencePiece encoding
        encoded_text = " ".join(self.spm_encoder.encode(sentence_text, out_type=str))
        return encoded_text

    def tokenize_file(self, inp_fname: Path, out_fname: Path) -> None:
        if not self.over_write and out_fname.exists():
            self.log(f"tokenized file {out_fname.name} already exists")
            return
        else:
            self.log(
                f"tokenizing {inp_fname.name}"
                + f"{' (de-escaped)' if self.descape else ''}"
                + f"{' (lower-cased)' if self.lower_case else ' (cased)'} "
                + f"(punctuation-normalization lang: {self.lang})"
            )

            with self.open(inp_fname, "rt") as file_in, open(
                out_fname, "w"
            ) as file_out:
                for line in file_in:
                    tokens = self.tokenize(line.strip())
                    file_out.write(tokens + "\n")

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

import os
import sys
import logging
from pathlib import Path
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
        self.moses_punct_normalizer = MosesPunctNormalizer(self.lang)
        self.moses_detokenizer = MosesDetokenizer()
        self.spm_encoder = spm.SentencePieceProcessor(model_file=self.spm_model)

    def tokenize(self, text: str) -> list[str]:
        # Preprocessing
        sentence_text = "".join(c for c in text if c.isprintable())
        sentence_text = self.moses_punct_normalizer.normalize(sentence_text)
        if self.descape:
            sentence_text = self.moses_detokenizer.unescape_xml(text=sentence_text)
        sentence_text = sentence_text.lower()

        # SentencePiece encoding
        spm_encoder = self.spm_encoder
        encoded_text = " ".join(spm_encoder.encode(sentence_text, out_type=str))

        return encoded_text

    def tokenize_file(self, inp_fname: Path, out_fname: Path, gzip: bool = False):
        if not os.path.isfile(out_fname):
            cat = "zcat " if gzip else "cat "
            if self.verbose:
                logger.info(
                    "SPM processing {} {} {}".format(
                        os.path.basename(inp_fname),
                        "(gzip)" if gzip else "",
                        "(de-escaped)" if self.descape else "",
                    )
                )

            with inp_fname.open("r", encoding="utf-8") as file:
                text = file.read()
            encoded_text = self.tokenize(text)

            with out_fname.open("w", encoding="utf-8") as file:
                file.write("".join(encoded_text))

        elif not self.over_write and self.verbose:
            logger.info(
                "SPM encoded file {} exists already".format(os.path.basename(out_fname))
            )

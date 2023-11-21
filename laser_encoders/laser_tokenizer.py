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

import gzip
import logging
import os
import re
import sys
from pathlib import Path
from typing import IO, List

import sentencepiece as spm
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from unicategories import categories

from laser_encoders.download_models import LaserModelDownloader
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE, SPM_LANGUAGE

SPACE_NORMALIZER = re.compile(r"\s+")
NON_PRINT_CHARS = set(c for c in categories["C"].characters())

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
        normalize_punct: bool = True,
    ):
        self.spm_model = spm_model
        self.lang = lang
        self.lower_case = lower_case
        self.descape = descape
        self.verbose = verbose
        self.over_write = over_write
        self.normalize_punct = normalize_punct

        assert spm_model.exists(), f"spm model file: {spm_model} does not exist"
        self.moses_punct_normalizer = MosesPunctNormalizer(self.lang, perl_parity=True)
        # add parity with MOSES release-4.0
        self.moses_punct_normalizer.substitutions[21] = ("‘", r'"')
        self.moses_punct_normalizer.substitutions[22] = ("‚", r'"')
        self.moses_detokenizer = MosesDetokenizer()
        self.spm_encoder = spm.SentencePieceProcessor(model_file=str(self.spm_model))

    def open(self, file: Path, mode: str, encoding="utf-8") -> IO:
        return (
            gzip.open(file, mode, encoding=encoding)
            if file.name.endswith(".gz")
            else open(file, mode, encoding=encoding)
        )

    def log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def tokenize(self, text: str) -> str:
        # Preprocessing
        sentence_text = "".join([c if c not in NON_PRINT_CHARS else " " for c in text])
        if self.normalize_punct:
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

    def __call__(self, text_or_batch):
        if isinstance(text_or_batch, str):
            return self.tokenize(text_or_batch)
        else:
            return self.tokenize_batch(text_or_batch)

    def tokenize_batch(self, batch: List[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in batch]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.spm_encoder.DecodeIds(ids) for ids in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        ids = []

        for token in tokens:
            # Apply the same tokenization logic as in _tokenize method
            tokens = SPACE_NORMALIZER.sub(" ", token).strip().split()

            # Initialize an empty tensor for this token's IDs
            token_ids = []

            for i, token in enumerate(tokens):
                token_id = self.spm_encoder.PieceToId(token)
                if token_id == 0:  # Handle out-of-vocabulary tokens
                    token_id = self.spm_encoder.PieceToId("<unk>")
                token_ids.append(token_id)

            # Append token IDs to the final IDs tensor
            ids.extend(token_ids)

        return ids


def initialize_tokenizer(lang: str = None, model_dir: str = None, laser: str = None):
    downloader = LaserModelDownloader(model_dir)
    if laser is not None:
        if laser == "laser3":
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            if lang in SPM_LANGUAGE:
                filename = f"laser3-{lang}.v1.spm"
            else:
                filename = "laser2.spm"
        elif laser == "laser2":
            filename = "laser2.spm"
        else:
            raise ValueError(
                f"Unsupported laser model: {laser}. Choose either laser2 or laser3."
            )
    else:
        if lang in LASER3_LANGUAGE:
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            if lang in SPM_LANGUAGE:
                filename = f"laser3-{lang}.v1.spm"
            else:
                filename = "laser2.spm"
        elif lang in LASER2_LANGUAGE:
            filename = "laser2.spm"
        else:
            raise ValueError(
                f"Unsupported language name: {lang}. Please specify a supported language name."
            )

    downloader.download(filename)
    model_path = os.path.join(downloader.model_dir, filename)
    return LaserTokenizer(spm_model=Path(model_path))

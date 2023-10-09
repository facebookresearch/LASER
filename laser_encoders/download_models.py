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
# -------------------------------------------------------
#
# This python script installs NLLB LASER2 and LASER3 sentence encoders from Amazon s3

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE, SPM_LANGUAGE
from laser_encoders.laser_tokenizer import LaserTokenizer
from laser_encoders.models import SentenceEncoder

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class LaserModelDownloader:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.expanduser("~/.cache/laser_encoders")
            os.makedirs(model_dir, exist_ok=True)

        self.model_dir = Path(model_dir)
        self.base_url = "https://dl.fbaipublicfiles.com/nllb/laser"

    def download(self, filename: str):
        url = os.path.join(self.base_url, filename)

        local_file_path = os.path.join(self.model_dir, filename)

        if os.path.exists(local_file_path):
            logger.info(f" - {filename} already downloaded")
        else:
            logger.info(f" - Downloading {filename}")

            tf = tempfile.NamedTemporaryFile(delete=False)
            temp_file_path = tf.name

            with tf:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("Content-Length", 0))
                progress_bar = tqdm(total=total_size, unit_scale=True, unit="B")

                for chunk in response.iter_content(chunk_size=1024):
                    tf.write(chunk)
                    progress_bar.update(len(chunk))
                progress_bar.close()

            os.rename(temp_file_path, local_file_path)

    def get_language_code(self, language_list: dict, lang: str) -> str:
        try:
            lang_3_4 = language_list[lang]
            if isinstance(lang_3_4, tuple):
                options = ", ".join(f"'{opt}'" for opt in lang_3_4)
                raise ValueError(
                    f"Language '{lang_3_4}' has multiple options: {options}. Please specify using --lang."
                )
            return lang_3_4
        except KeyError:
            raise ValueError(
                f"language name: {lang} not found in language list. Specify a supported language name"
            )

    def download_laser2(self):
        self.download("laser2.pt")
        self.download("laser2.spm")
        self.download("laser2.cvocab")

    def download_laser3(self, lang: str, spm: bool = False):
        lang = self.get_language_code(LASER3_LANGUAGE, lang)
        self.download(f"laser3-{lang}.v1.pt")
        if spm:
            if lang in SPM_LANGUAGE:
                self.download(f"laser3-{lang}.v1.spm")
                self.download(f"laser3-{lang}.v1.cvocab")
            else:
                self.download(f"laser2.spm")
                self.download(f"laser2.cvocab")

    def main(self, args):
        if args.laser:
            if args.laser == "laser2":
                self.download_laser2()
            elif args.laser == "laser3":
                self.download_laser3(lang=args.lang, spm=args.spm)
            else:
                raise ValueError(
                    f"Unsupported laser model: {args.laser}. Choose either laser2 or laser3."
                )
        else:
            if args.lang in LASER3_LANGUAGE:
                self.download_laser3(lang=args.lang, spm=args.spm)
            elif args.lang in LASER2_LANGUAGE:
                self.download_laser2()
            else:
                raise ValueError(
                    f"Unsupported language name: {args.lang}. Please specify a supported language name using --lang."
                )


def initialize_encoder(
    lang: str = None,
    model_dir: str = None,
    spm: bool = True,
    laser: str = None,
    tokenize: bool = False,
):
    downloader = LaserModelDownloader(model_dir)
    if laser is not None:
        if laser == "laser3":
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            downloader.download_laser3(lang=lang, spm=spm)
            file_path = f"laser3-{lang}.v1"
        elif laser == "laser2":
            downloader.download_laser2()
            file_path = "laser2"
        else:
            raise ValueError(
                f"Unsupported laser model: {laser}. Choose either laser2 or laser3."
            )
    else:
        lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
        if lang in LASER3_LANGUAGE:
            downloader.download_laser3(lang=lang, spm=spm)
            file_path = f"laser3-{lang}.v1"
        elif lang in LASER2_LANGUAGE:
            downloader.download_laser2()
            file_path = "laser2"
        else:
            raise ValueError(
                f"Unsupported language name: {lang}. Please specify a supported language name."
            )

    model_dir = downloader.model_dir
    model_path = os.path.join(model_dir, f"{file_path}.pt")
    spm_vocab = os.path.join(model_dir, f"{file_path}.cvocab")
    spm_model = None
    if not os.path.exists(spm_vocab):
        # if there is no cvocab for the laser3 lang use laser2 cvocab
        spm_vocab = os.path.join(model_dir, "laser2.cvocab")
    if tokenize:
        spm_model = os.path.join(model_dir, f"{file_path}.spm")
        if not os.path.exists(spm_model):
            spm_model = os.path.join(model_dir, "laser2.spm")

    return SentenceEncoder(
        model_path=model_path, spm_vocab=spm_vocab, spm_model=spm_model
    )


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
        if lang in LASER3_LANGUAGE or lang in LASER2_LANGUAGE:
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            if lang in SPM_LANGUAGE:
                filename = f"laser3-{lang}.v1.spm"
            else:
                filename = "laser2.spm"
        else:
            raise ValueError(
                f"Unsupported language name: {lang}. Please specify a supported language name."
            )

    downloader.download(filename)
    model_path = os.path.join(downloader.model_dir, filename)
    return LaserTokenizer(spm_model=Path(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Download Laser models")
    parser.add_argument(
        "--laser",
        type=str,
        help="Laser model to download",
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="The language name in FLORES200 format",
    )
    parser.add_argument(
        "--spm",
        action="store_false",
        help="Do not download the SPM model?",
    )
    parser.add_argument(
        "--model-dir", type=str, help="The directory to download the models to"
    )
    args = parser.parse_args()
    downloader = LaserModelDownloader(args.model_dir)
    downloader.main(args)

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
# default to download to current directory

import argparse
import logging
import os
import sys
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
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.base_url = "https://dl.fbaipublicfiles.com/nllb/laser"

    def download(self, filename: str):
        url = os.path.join(self.base_url, filename)
        local_file_path = self.model_dir / filename

        if local_file_path.exists():
            logger.info(f" - {filename} already downloaded")
        else:
            logger.info(f" - Downloading {filename}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("Content-Length", 0))
            progress_bar = tqdm(total=total_size, unit="KB")
            with open(local_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()

    def get_language_code(self, language_list: dict, lang: str) -> str:
        lang_3_4 = language_list[lang]
        if isinstance(lang_3_4, tuple):
            options = ", ".join(f"'{opt}'" for opt in lang_3_4)
            raise ValueError(
                f"Language '{lang_3_4}' has multiple options: {options}. Please specify using --lang."
            )
        return lang_3_4

    def download_laser2(self, lang: str):
        if self.get_language_code(LASER2_LANGUAGE, lang):
            self.download("laser2.pt")
            self.download("laser2.spm")
            self.download("laser2.cvocab")

    def download_laser3(self, lang: str, version: str = "v1", spm: bool = False):
        lang = self.get_language_code(LASER3_LANGUAGE, lang)
        self.download(f"laser3-{lang}.{version}.pt")
        if spm:
            if lang in SPM_LANGUAGE:
                self.download(f"laser3-{lang}.{version}.spm")
                self.download(f"laser3-{lang}.{version}.cvocab")
            else:
                self.download(f"laser2.spm")
                self.download(f"laser2.cvocab")

    def main(self, args):
        if args.laser:
            if args.laser == "laser2":
                self.download_laser2(args.lang)
            elif args.laser == "laser3":
                self.download_laser3(lang=args.lang, version=args.version, spm=args.spm)
        else:
            if args.lang in LASER3_LANGUAGE:
                self.download_laser3(lang=args.lang, version=args.version, spm=args.spm)
            elif args.lang in LASER2_LANGUAGE:
                self.download_laser2(args.lang)
            else:
                raise ValueError(
                    f"Unsupported language name: {args.lang}. Please specify a supported language name using --lang."
                )


def initialize_encoder(
    lang: str,
    model_dir: str = None,
    version: str = "v1",
    spm: bool = True,
    laser: str = None,
):
    if model_dir is None:
        model_dir = os.path.expanduser("~/.cache/laser_encoders")

    downloader = LaserModelDownloader(model_dir)
    if laser is not None:
        if laser == "laser3":
            downloader.download_laser3(lang=lang, version=version, spm=spm)
            file_path = f"laser3-{lang}.{version}"
        elif laser == "laser2":
            downloader.download_laser2(lang)
            file_path = "laser2"
        else:
            raise ValueError(f"Unsupported laser model: {laser}.")
    else:
        if lang in LASER3_LANGUAGE:
            downloader.download_laser3(lang=lang, version=version, spm=spm)
            file_path = f"laser3-{lang}.{version}"
        elif lang in LASER2_LANGUAGE:
            downloader.download_laser2(lang)
            file_path = "laser2"
        else:
            raise ValueError(
                f"Unsupported language name: {lang}. Please specify a supported language name."
            )
    model_path = f"{model_dir}/{file_path}.pt"
    spm_path = f"{model_dir}/{file_path}.cvocab"
    if not os.path.exists(spm_path):
        # if there is no cvocab for the laser3 lang use laser2 cvocab
        spm_path = os.path.join(model_dir, "laser2.cvocab")
    return SentenceEncoder(model_path=model_path, spm_vocab=spm_path)


def initialize_tokenizer(
    lang: str, model_dir: str = None, version: str = "v1", laser: str = None
):
    if model_dir is None:
        model_dir = os.path.expanduser("~/.cache/laser_encoders")

    downloader = LaserModelDownloader(model_dir)
    if laser is not None:
        if laser == "laser3":
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            filename = f"laser3-{lang}.{version}.spm"
        elif laser == "laser2":
            filename = "laser2.spm"
        else:
            raise ValueError(f"Unsupported laser model: {laser}.")
    else:
        if lang in LASER3_LANGUAGE or lang in LASER2_LANGUAGE:
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            if lang in SPM_LANGUAGE:
                filename = f"laser3-{lang}.{version}.spm"
            else:
                filename = "laser2.spm"
        else:
            raise ValueError(
                f"Unsupported language name: {lang}. Please specify a supported language name."
            )

    downloader.download(filename)
    model_path = model_dir + filename
    return LaserTokenizer(spm_model=Path(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Download Laser models")
    parser.add_argument(
        "--laser",
        type=str,
        required=False,
        help="Laser model to download",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="The language name in FLORES200 format",
    )
    parser.add_argument(
        "--version", type=str, default="v1", help="The encoder model version"
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
    model_dir = args.model_dir or os.path.expanduser("~/.cache/laser_encoders")
    downloader = LaserModelDownloader(model_dir)
    downloader.main(args)

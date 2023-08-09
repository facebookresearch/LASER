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
from language_list import LASER2_LANGUAGE, LASER3_LANGUAGE, SPM_LANGUAGE
from tqdm import tqdm

assert os.environ.get("LASER"), "Please set the environment variable LASER"
LASER = os.environ["LASER"]
sys.path.append(LASER)

from laser_encoders.laser_tokenizer import LaserTokenizer
from laser_encoders.models import SentenceEncoder

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("preprocess")


class LaserModelDownloader:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.url = "https://dl.fbaipublicfiles.com/nllb/laser"

    def download(self, url: str):
        if os.path.exists(os.path.join(self.model_dir, os.path.basename(url))):
            logger.info(f" - {os.path.basename(url)} already downloaded")
        else:
            logger.info(f" - Downloading {os.path.basename(url)}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("Content-Length", 0))
            progress_bar = tqdm(total=total_size, unit="KB")
            with open(os.path.join(self.model_dir, os.path.basename(url)), "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()

    def download_laser2(self):
        self.download(f"{self.url}/laser2.pt")
        self.download(f"{self.url}/laser2.spm")
        self.download(f"{self.url}/laser2.cvocab")

    def download_laser3(self, lang: str, version: str = "v1", spm: bool = False):
        lang_3_4 = LASER3_LANGUAGE[lang]
        if isinstance(lang_3_4, tuple):
            options = ", ".join(f"'{opt}'" for opt in lang_3_4)
            logger.info(
                f"Language '{lang_3_4}' has multiple options: {options}. Please specify using --lang-model."
            )
            return

        self.download(f"{self.url}/laser3-{lang_3_4}.{version}.pt")
        if spm:
            if lang_3_4 in SPM_LANGUAGE:
                self.download(f"{self.url}/laser3-{lang_3_4}.{version}.spm")
                self.download(f"{self.url}/laser3-{lang_3_4}.{version}.cvocab")
            else:
                self.download(f"{self.url}/laser2.spm")
                self.download(f"{self.url}/laser2.cvocab")

    def main(self, args):
        if args.lang_model in LASER2_LANGUAGE:
            self.download_laser2()
        elif args.lang_model in LASER3_LANGUAGE:
            self.download_laser3(
                lang=args.lang_model, version=args.version, spm=args.spm
            )
        else:
            raise ValueError(
                f"Unsupported language name: {args.lang_model}. Please specify a supported language name using --lang-model."
            )


def download_model(model_dir, version, lang_model, spm):
    if model_dir is None:
        model_dir = os.path.expanduser("~/.cache/laser_encoders")
    file_path = ""
    downloader = LaserModelDownloader(model_dir)
    if lang_model in LASER2_LANGUAGE:
        downloader.download_laser2()
        file_path = model_dir + "/laser2"
    elif lang_model in LASER3_LANGUAGE:
        downloader.download_laser3(lang=lang_model, version=version, spm=spm)
        file_path = f"{model_dir}/laser3-{lang_model}.{version}"
    else:
        raise ValueError(
            f"Unsupported language name: {lang_model}. Please specify a supported language name using --lang-model."
        )
    return file_path


def initialize_encoder(lang_model, model_dir=None, version="v1", spm=False):
    file_path = download_model(model_dir, version, lang_model, spm)
    model_path = f"{file_path}.pt"
    spm_path = f"{file_path}.cvocab"
    if not os.path.exists(spm_path):
        model_dir, _ = os.path.split(spm_path)
        spm_path = f'{model_dir}/laser2.cvocab'
    return SentenceEncoder(model_path=model_path, spm_vocab=spm_path)


def initialize_tokenizer(lang_model, model_dir=None, version="v1", spm=False):
    file_path = download_model(model_dir, version, lang_model, spm)
    model_path = f"{file_path}.spm"
    return LaserTokenizer(spm_model=Path(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Download Laser models")
    parser.add_argument(
        "--lang-model",
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

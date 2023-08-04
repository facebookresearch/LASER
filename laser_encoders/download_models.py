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
#-------------------------------------------------------
#
# This python script installs NLLB LASER2 and LASER3 sentence encoders from Amazon s3

# default to download to current directory

import argparse
import logging
import os
import sys

import requests
from language_list import LANGUAGE_MAPPING, SPM_LANGUAGE

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("preprocess")

class LaserModelDownloader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.url = "https://dl.fbaipublicfiles.com/nllb/laser/"

    def download(self, url):
        if os.path.exists(os.path.join(self.model_dir, os.path.basename(url))):
            logger.info(f" - {os.path.basename(url)} already downloaded")
        else:
            logger.info(f" - Downloading {os.path.basename(url)}")
            response = requests.get(url, stream=True)
            with open(os.path.join(self.model_dir, os.path.basename(url)), "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

    def download_laser2(self):
        self.download(f"{self.url}/laser2.pt")
        self.download(f"{self.url}/laser2.spm")
        self.download(f"{self.url}/laser2.cvocab")

    def download_laser3(self, lang, version="v1", spm=False):
        if lang not in LANGUAGE_MAPPING:
            logger.info(f"Unsupported language name: {lang}. Please specify a supported language name.")
            return
    
        if len(LANGUAGE_MAPPING[lang]) > 1:
            options = ', '.join(f"'{opt}'" for opt in LANGUAGE_MAPPING[lang])
            logger.info(f"Language '{lang}' has multiple options: {options}. Please specify.")
            return

        lang = LANGUAGE_MAPPING[lang]
        file_name = f"{self.url}/laser3-{lang}.{version}.pt"
        self.download(file_name)
        if spm:
            if lang in SPM_LANGUAGE:
                file_name = f"{self.url}/laser3-{lang}.{version}.spm"
                self.download(file_name)
                file_name = f"{self.url}/laser3-{lang}.{version}.cvocab"
                self.download(file_name)
            else:
                self.download(f"{self.url}/laser2.spm")
                self.download(f"{self.url}/laser2.cvocab")

    def main(self, args):
        if args.laser == "laser2":
            self.download_laser2()
        elif args.laser == "laser3":
            self.download_laser3(lang=args.lang, version=args.version, spm=args.spm)
        else:
            logger.info("Please specify --laser. either laeser2 or laser3")
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Download Laser models")
    parser.add_argument("--laser", help="LASER model to download")
    parser.add_argument("--lang", type=str, required=True, help="The language name in FLORES200 format")
    parser.add_argument("--version", type=str, help="The encoder model version")
    parser.add_argument("--spm", help="Download the SPM model as well?")
    parser.add_argument("--model_dir", type=str, help="The directory to download the models to")
    args = parser.parse_args()
    model_dir = args.model_dir or os.path.expanduser("~/.cache/laser_encoders")
    downloader = LaserModelDownloader(model_dir)
    downloader.main(args)
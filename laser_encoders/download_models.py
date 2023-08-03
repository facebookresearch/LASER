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

import os
import requests
import logging

logger = logging.getLogger(__name__)

class LaserModelDownloader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.url = "https://dl.fbaipublicfiles.com/nllb/laser"

    def download(self):
        if os.path.exists(os.path.join(self.model_dir, os.path.basename(self.url))):
            logger.info(f" - {os.path.basename(self.url)} already downloaded")
        else:
            logger.info(f" - Downloading {os.path.basename(self.url)}")
            response = requests.get(self.url, stream=True)
            with open(os.path.join(self.model_dir, os.path.basename(self.url)), "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

    def download_laser2(self):
        self.download(f"{self.url}/laser2.pt")
        self.download(f"{self.url}/laser2.spm")
        self.download(f"{self.url}laser2.cvocab")

    def download_laser3(self, lang, version="v1", spm=False):
        file_name = f"laser3-{lang}.{version}.pt"
        self.download(file_name)
        if spm:
            file_name = f"laser3-{lang}.{version}.spm"
            self.download(file_name)
            file_name = f"laser3-{lang}.{version}.cvocab"
            self.download(file_name)

    def main(args):
        model_dir = args.model_dir or os.path.expanduser("~/.cache/laser_encoders")
        downloader = LaserModelDownloader(model_dir)

        if args.laser == "laser2":
            downloader.download_laser2()
        elif args.laser == "laser3":
            lang = args.lang
            version = args.version
            spm = args.spm
            downloader.download_laser3(lang, version, spm)
        else:
            raise ValueError("Please specify either --laser. either laeser2 or laser3")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LASER: Embed sentences")
    parser.add_argument("--laser", action="store_true", help="LASER model to download")
    parser.add_argument("--lang", type=str, help="The language name in FLORES200 format")
    parser.add_argument("--version", type=str, help="The encoder model version")
    parser.add_argument("--spm", action="store_true", help="Download the SPM model as well")
    args = parser.parse_args()
    laser_downloader = LaserModelDownloader(version=1, model_dir=os.path.expanduser("~/.cache/laser_encoders"))
    laser_downloader.main(args)
import os
import tempfile

import pytest

from laser_encoders.download_models import LaserModelDownloader
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.laser_tokenizer import initialize_tokenizer
from laser_encoders.models import initialize_encoder


def test_validate_achnese_models_and_tokenize_laser3(lang="acehnese"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        downloader.download_laser3(lang)
        encoder = initialize_encoder(lang, model_dir=tmp_dir)
        tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

        # Test tokenization with a sample sentence
        tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")


def test_validate_english_models_and_tokenize_laser2(lang="english"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        downloader.download_laser2()

        encoder = initialize_encoder(lang, model_dir=tmp_dir)
        tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

        # Test tokenization with a sample sentence
        tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")


def test_validate_kashmiri_models_and_tokenize_laser3(lang="kas"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        with pytest.raises(ValueError):
            downloader.download_laser3(lang)

            encoder = initialize_encoder(lang, model_dir=tmp_dir)
            tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

            # Test tokenization with a sample sentence
            tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")

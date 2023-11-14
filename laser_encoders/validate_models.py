import os
import tempfile

import pytest

from laser_encoders.download_models import LaserModelDownloader
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.laser_tokenizer import initialize_tokenizer
from laser_encoders.models import initialize_encoder


@pytest.mark.slow
@pytest.mark.parametrize("lang", LASER3_LANGUAGE)
def test_validate_language_models_and_tokenize_laser3(lang):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        if lang in ["kashmiri", "kas", "central kanuri", "knc"]:
            with pytest.raises(ValueError) as excinfo:
                downloader.download_laser3(lang)
            assert "ValueError" in str(excinfo.value)
            print(f"{lang} language model raised a ValueError as expected.")
        else:
            downloader.download_laser3(lang)
            encoder = initialize_encoder(lang, model_dir=tmp_dir)
            tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

            # Test tokenization with a sample sentence
            tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")


@pytest.mark.slow
@pytest.mark.parametrize("lang", LASER2_LANGUAGE)
def test_validate_language_models_and_tokenize_laser2(lang):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        downloader.download_laser2()

        encoder = initialize_encoder(lang, model_dir=tmp_dir)
        tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

        # Test tokenization with a sample sentence
        tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")


class MockLaserModelDownloader(LaserModelDownloader):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def download_laser3(self, lang):
        lang = self.get_language_code(LASER3_LANGUAGE, lang)
        file_path = os.path.join(self.model_dir, f"laser3-{lang}.v1.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find {file_path}.")

    def download_laser2(self):
        files = ["laser2.pt", "laser2.spm", "laser2.cvocab"]
        for file_name in files:
            file_path = os.path.join(self.model_dir, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Could not find {file_path}.")


CACHE_DIR = "/home/user/.cache/models"  # Change this to the desired cache directory

# This uses the mock downloader
@pytest.mark.slow
@pytest.mark.parametrize("lang", LASER3_LANGUAGE)
def test_validate_language_models_and_tokenize_mock_laser3(lang):
    downloader = MockLaserModelDownloader(model_dir=CACHE_DIR)

    try:
        downloader.download_laser3(lang)
    except FileNotFoundError as e:
        raise pytest.error(str(e))

    encoder = initialize_encoder(lang, model_dir=CACHE_DIR)
    tokenizer = initialize_tokenizer(lang, model_dir=CACHE_DIR)

    tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")


# This uses the mock downloader
@pytest.mark.slow
@pytest.mark.parametrize("lang", LASER2_LANGUAGE)
def test_validate_language_models_and_tokenize_mock_laser2(lang):
    downloader = MockLaserModelDownloader(model_dir=CACHE_DIR)

    try:
        downloader.download_laser2()
    except FileNotFoundError as e:
        raise pytest.error(str(e))

    encoder = initialize_encoder(lang, model_dir=CACHE_DIR)
    tokenizer = initialize_tokenizer(lang, model_dir=CACHE_DIR)

    tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} model validated successfully")

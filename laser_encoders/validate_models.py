import os
import tempfile
import pytest

from laser_encoders.download_models import LaserModelDownloader
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.laser_tokenizer import initialize_tokenizer
from laser_encoders.models import initialize_encoder


@pytest.mark.parametrize("lang", LASER3_LANGUAGE)
def test_validate_language_models_and_tokenize_laser3(lang):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        downloader.download_laser3(lang)

        encoder = initialize_encoder(lang, model_dir=tmp_dir)
        tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

        # Test tokenization with a sample sentence
        tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} language model validated and deleted successfully.")

@pytest.mark.parametrize("lang", LASER2_LANGUAGE)
def test_validate_language_models_and_tokenize_laser2(lang):
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory for {lang}", tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)
        downloader.download_laser2()

        encoder = initialize_encoder(lang, model_dir=tmp_dir, laser="laser2")
        tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)

        # Test tokenization with a sample sentence
        tokenized = tokenizer.tokenize("This is a sample sentence.")

    print(f"{lang} language model validated and deleted successfully.")

class MockLaserModelDownloader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
    
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

    def download_laser3(self, lang):
        lang = self.get_language_code(LASER3_LANGUAGE, lang)
        file_path = os.path.join(self.model_dir, f"laser3-{lang}.v1.pt")
        if os.path.exists(file_path):
            return False
        else:
            return True

    def download_laser2(self):
        files = ["laser2.pt", "laser2.spm", "laser2.cvocab"]
        for file_name in files:
            file_path = os.path.join(self.model_dir, file_name)
            if os.path.exists(file_path):
                return False
            else:
                return True

CACHE_DIR = "/home/user/.cache/models"  # Change this to the desired cache directory

# This uses the mock downloader
@pytest.mark.parametrize("lang", LASER3_LANGUAGE)
def test_validate_language_models_and_tokenize_mock_laser3(lang):
    downloader = MockLaserModelDownloader(model_dir=CACHE_DIR)
    err = downloader.download_laser3(lang)
    if err == True:
        raise pytest.error(f"Skipping test for {lang} language.")

    encoder = initialize_encoder(lang, model_dir=CACHE_DIR)
    tokenizer = initialize_tokenizer(lang, model_dir=CACHE_DIR)

    tokenized = tokenizer.tokenize("This is a sample sentence.")

# This uses the mock downloader
@pytest.mark.parametrize("lang", LASER2_LANGUAGE)
def test_validate_language_models_and_tokenize_mock_laser2(lang):
    downloader = LaserModelDownloader(model_dir=CACHE_DIR)
    err = downloader.download_laser2()
    if err == True:
        raise pytest.error()

    encoder = initialize_encoder(lang, model_dir=CACHE_DIR, laser="laser2")
    tokenizer = initialize_tokenizer(lang, model_dir=CACHE_DIR)

    tokenized = tokenizer.tokenize("This is a sample sentence.")

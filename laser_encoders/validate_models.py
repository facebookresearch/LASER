import os
import tempfile
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.download_models import LaserModelDownloader
from laser_encoders.models import initialize_encoder
from laser_encoders.laser_tokenizer import initialize_tokenizer

def validate_language_models_and_tokenize():
    with tempfile.TemporaryDirectory() as tmp_dir:
        print('Created temporary directory', tmp_dir)

        downloader = LaserModelDownloader(model_dir=tmp_dir)

        for lang in LASER3_LANGUAGE:
            # Use the downloader to download the model
            downloader.download_laser3(lang)
            encoder = initialize_encoder(lang, model_dir=tmp_dir)
            tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)
            # Test tokenization with a sample sentence
            tokenized = tokenizer.tokenize("This is a sample sentence.")

        for lang in LASER2_LANGUAGE:
            # Use the downloader to download the model
            downloader.download_laser2()
            encoder = initialize_encoder(lang, model_dir=tmp_dir, laser="laser2")
            tokenizer = initialize_tokenizer(lang, model_dir=tmp_dir)
            # Test tokenization with a sample sentence
            tokenized = tokenizer.tokenize("This is a sample sentence.")

    print("All language models validated and deleted successfully.")

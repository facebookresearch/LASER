import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
import urllib.request
import sentencepiece as spm

from laser_tokenizer import LaserTokenizer
import logging
import sys



@pytest.fixture
def tokenizer():
    with NamedTemporaryFile(delete=False) as f:
        with urllib.request.urlopen('https://dl.fbaipublicfiles.com/nllb/laser/laser2.spm') as response:
            f.write(response.read())
    f.close()  # Manually close the file
    return LaserTokenizer(spm_model=Path(f.name))



def test_tokenize(tokenizer):
    test_data = "This is a test sentence."
    expected_output = "▁this ▁is ▁a ▁test ▁sent ence ."
    assert tokenizer.tokenize(test_data) == expected_output

def test_file_does_not_exist(tokenizer):
    os.remove(tokenizer.spm_model)
    assert not os.path.exists(tokenizer.spm_model.name)

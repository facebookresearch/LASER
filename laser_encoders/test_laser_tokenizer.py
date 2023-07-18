import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
import urllib.request

from laser_tokenizer import LaserTokenizer


@pytest.fixture
def tokenizer():
    with NamedTemporaryFile() as f:
        with urllib.request.urlopen(
            "https://dl.fbaipublicfiles.com/nllb/laser/laser2.spm"
        ) as response:
            f.write(response.read())
        return LaserTokenizer(spm_model=Path(f.name))

def test_tokenize(tokenizer):
    test_data = "This is a test sentence."
    expected_output = "▁this ▁is ▁a ▁test ▁sent ence ."
    assert tokenizer.tokenize(test_data) == expected_output


def test_normalization(tokenizer):
    test_data = "Hello!!! How are you??? I'm doing great."
    expected_output = "▁hel lo !!! ▁how ▁are ▁you ??? ▁i ' m ▁do ing ▁great ."
    assert tokenizer.tokenize(test_data) == expected_output


def test_descape(tokenizer):
    test_data = "I &lt;3 Apple &amp; Carrots!"
    expected_output = "▁i ▁<3 ▁app le ▁& ▁car ro ts !"
    tokenizer.descape = True
    assert tokenizer.tokenize(test_data) == expected_output


def test_lowercase(tokenizer):
    test_data = "THIS OUTPUT MUST BE UPPERCASE"
    expected_output = "▁TH IS ▁ OU TP UT ▁ MU ST ▁BE ▁ UP PER CA SE"
    tokenizer.lower_case = False
    assert tokenizer.tokenize(test_data) == expected_output

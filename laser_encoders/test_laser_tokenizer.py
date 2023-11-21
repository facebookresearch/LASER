#!/usr/bin/python3
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
# --------------------------------------------------------
# Tests for LaserTokenizer

import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
import pytest

from laser_encoders import (
    LaserEncoderPipeline,
    initialize_encoder,
    initialize_tokenizer,
)


@pytest.fixture
def tokenizer(tmp_path: Path):
    tokenizer_instance = initialize_tokenizer(model_dir=tmp_path, laser="laser2")
    return tokenizer_instance


@pytest.fixture
def input_text() -> str:
    return "This is a test sentence."


@pytest.fixture
def test_readme_params() -> dict:
    return {
        "lang": "igbo",
        "input_sentences": ["nnọọ, kedu ka ị mere"],
        "expected_embedding_shape": (1, 1024),
        "expected_array": [
            0.3807628,
            -0.27941525,
            -0.17819545,
            0.44144684,
            -0.38985375,
            0.04719935,
            0.20238206,
            -0.03934783,
            0.0118901,
            0.28986093,
        ],
    }


def test_tokenize(tokenizer, input_text: str):
    expected_output = "▁this ▁is ▁a ▁test ▁sent ence ."
    assert tokenizer.tokenize(input_text) == expected_output


def test_tokenizer_call_method(tokenizer, input_text: str):
    single_string = "This is a test sentence."
    expected_output = "▁this ▁is ▁a ▁test ▁sent ence ."
    assert tokenizer(single_string) == expected_output

    list_of_strings = ["This is a test sentence.", "This is another test sentence."]
    expected_output = [
        "▁this ▁is ▁a ▁test ▁sent ence .",
        "▁this ▁is ▁another ▁test ▁sent ence .",
    ]
    assert tokenizer(list_of_strings) == expected_output


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


def test_is_printable(tokenizer):
    test_data = "Hello, \tWorld! ABC\x1f123"
    expected_output = "▁hel lo , ▁world ! ▁ab c ▁12 3"
    assert tokenizer.tokenize(test_data) == expected_output


def test_tokenize_file(tokenizer, input_text: str):
    with TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "input.txt")
        output_file = os.path.join(temp_dir, "output.txt")

        with open(input_file, "w") as file:
            file.write(input_text)

        tokenizer.tokenize_file(
            inp_fname=Path(input_file),
            out_fname=Path(output_file),
        )

        with open(output_file, "r") as file:
            output = file.read().strip()

        expected_output = "▁this ▁is ▁a ▁test ▁sent ence ."
        assert output == expected_output


def test_tokenize_file_overwrite(tokenizer, input_text: str):
    with TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "input.txt")
        output_file = os.path.join(temp_dir, "output.txt")

        with open(input_file, "w") as file:
            file.write(input_text)

        with open(output_file, "w") as file:
            file.write("Existing output")

        # Test when over_write is False
        tokenizer.over_write = False
        tokenizer.tokenize_file(
            inp_fname=Path(input_file),
            out_fname=Path(output_file),
        )

        with open(output_file, "r") as file:
            output = file.read().strip()

        assert output == "Existing output"

        # Test when over_write is True
        tokenizer.over_write = True
        tokenizer.tokenize_file(
            inp_fname=Path(input_file),
            out_fname=Path(output_file),
        )

        with open(output_file, "r") as file:
            output = file.read().strip()

        expected_output = "▁this ▁is ▁a ▁test ▁sent ence ."
        assert output == expected_output


@pytest.mark.parametrize(
    "laser, expected_array, lang",
    [
        (
            "laser2",
            [
                1.042462512850761414e-02,
                6.325428839772939682e-03,
                -3.032622225873637944e-05,
                9.033476933836936951e-03,
                2.937933895736932755e-04,
                4.489220678806304932e-03,
                2.334521152079105377e-03,
                -9.427300537936389446e-04,
                -1.571535394759848714e-04,
                2.095808042213320732e-03,
            ],
            None,
        ),
        (
            "laser3",
            [
                3.038274645805358887e-01,
                4.151830971240997314e-01,
                -2.458990514278411865e-01,
                3.153458833694458008e-01,
                -5.153598189353942871e-01,
                -6.035178527235984802e-02,
                2.210616767406463623e-01,
                -2.701394855976104736e-01,
                -4.902199506759643555e-01,
                -3.126966953277587891e-02,
            ],
            "zul_Latn",
        ),
    ],
)
def test_sentence_encoder(
    tmp_path: Path,
    tokenizer,
    laser: str,
    expected_array: List,
    lang: str,
    input_text: str,
):
    sentence_encoder = initialize_encoder(model_dir=tmp_path, laser=laser, lang=lang)
    tokenized_text = tokenizer.tokenize(input_text)
    sentence_embedding = sentence_encoder.encode_sentences([tokenized_text])

    assert isinstance(sentence_embedding, np.ndarray)
    assert sentence_embedding.shape == (1, 1024)
    assert np.allclose(expected_array, sentence_embedding[:, :10], atol=1e-3)


def test_laser_encoder_pipeline(tmp_path: Path, test_readme_params: dict):
    lang = test_readme_params["lang"]
    input_sentences = test_readme_params["input_sentences"]
    expected_embedding_shape = test_readme_params["expected_embedding_shape"]
    expected_array = test_readme_params["expected_array"]

    encoder = LaserEncoderPipeline(model_dir=tmp_path, lang=lang)
    embeddings = encoder.encode_sentences(input_sentences)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == expected_embedding_shape
    assert np.allclose(expected_array, embeddings[:, :10], atol=1e-3)


def test_separate_initialization_and_encoding(
    tmp_path, tokenizer, test_readme_params: dict
):
    lang = test_readme_params["lang"]
    input_sentences = test_readme_params["input_sentences"]
    expected_embedding_shape = test_readme_params["expected_embedding_shape"]
    expected_array = test_readme_params["expected_array"]

    tokenized_sentence = tokenizer.tokenize(input_sentences[0])
    sentence_encoder = initialize_encoder(model_dir=tmp_path, lang=lang)

    # Encode tokenized sentences into embeddings
    embeddings = sentence_encoder.encode_sentences([tokenized_sentence])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == expected_embedding_shape
    assert np.allclose(expected_array, embeddings[:, :10], atol=1e-3)


def test_encoder_normalization(tmp_path: Path, test_readme_params: dict):
    lang = test_readme_params["lang"]
    input_sentences = test_readme_params["input_sentences"]

    encoder = LaserEncoderPipeline(model_dir=tmp_path, lang=lang)
    normalized_embeddings = encoder.encode_sentences(
        input_sentences, normalize_embeddings=True
    )
    norm = np.linalg.norm(normalized_embeddings[0])

    assert np.allclose(norm, 1.0, atol=1e-3)


def test_encoder_default_behaviour(tmp_path: Path, test_readme_params: dict):
    lang = test_readme_params["lang"]
    input_sentences = test_readme_params["input_sentences"]

    encoder = LaserEncoderPipeline(model_dir=tmp_path, lang=lang)
    default_embeddings = encoder.encode_sentences(input_sentences)
    non_normalized_embeddings = encoder.encode_sentences(
        input_sentences, normalize_embeddings=False
    )

    assert np.allclose(default_embeddings, non_normalized_embeddings)


def test_encoder_non_normalization(tmp_path: Path, test_readme_params: dict):
    lang = test_readme_params["lang"]
    input_sentences = test_readme_params["input_sentences"]

    encoder = LaserEncoderPipeline(model_dir=tmp_path, lang=lang)
    non_normalized_embeddings = encoder.encode_sentences(
        input_sentences, normalize_embeddings=False
    )
    norm = np.linalg.norm(non_normalized_embeddings[0])

    assert not np.isclose(norm, 1)


def test_optional_lang_with_laser2(tmp_path: Path):
    with pytest.warns(
        UserWarning,
        match="The 'lang' parameter is optional when using 'laser2'. It will be ignored.",
    ):
        encoder = LaserEncoderPipeline(lang="en", laser="laser2", model_dir=tmp_path)


def test_required_lang_with_laser3(tmp_path: Path):
    with pytest.raises(
        ValueError, match="For 'laser3', the 'lang' parameter is required."
    ):
        encoder = LaserEncoderPipeline(laser="laser3", model_dir=tmp_path)


def test_missing_lang_and_laser(tmp_path: Path):
    with pytest.raises(
        ValueError, match="Either 'laser' or 'lang' should be provided."
    ):
        encoder = LaserEncoderPipeline(model_dir=tmp_path)

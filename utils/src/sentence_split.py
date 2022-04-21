# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
from pathlib import Path

# Indicp NLP
from indicnlp import common as indic_common
from indicnlp import loader as indic_loader
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import sentence_tokenize as indic_sent_tok

# --- sentence splitters
# Moses-style
from sentence_splitter import SentenceSplitter

INDIC_NLP_RESOURCES = None  # apparently not needed for splitting and normalization
from botok.tokenizers import sentencetokenizer as bod_sent_tok
from khmernltk import sentence_tokenize as khm_sent_tok

# pythainlp for Thai
# Seahorse for Indonesian, Thai, Vietnamese
# botok for tibetan
# Spacy for
# various tool-kits
from laonlp.tokenize import sent_tokenize as lao_sent_tok

logger = logging.getLogger("sentence_split")


# ----------------------------------
# Supported tokenization algorithms
# List of supported languages and mapping ISO3 - > ISO2

LANGS_MOSES = {
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "nld": "nl",
    "eng": "en",
    "fin": "fi",
    "fra": "fr",
    "deu": "de",
    "ell": "el",
    "hun": "hu",
    "isl": "is",
    "ita": "it",
    "lav": "lv",
    "lit": "lt",
    "nob": "no",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "swe": "sv",
    "tur": "tr",
}

LANGS_LAONLP = {"lao": "lao"}
LANGS_KHMER = {"khm": "khm"}
LANGS_BODNLP = {
    "bod": "bod",
    "dzo": "dzo",
}  # languages with tibetan script

# ----------------------------------------------
LANGS_INDIC = {
    "asm": "as",
    "ben": "bn",
    "brx": "bD",
    "gom": "xx",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "kok": "kK",
    "mni": "bn",  # our meitei is in bengali script, so swapped it to bengali here
    "mal": "ml",
    "mar": "mr",
    "npi": "ne",
    "ory": "or",
    "pan": "pa",
    "san": "sa",
    "snd": "sd",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

# ----------------------------------------------
LANGS_GEEZ = {"amh": "amh", "tir": "tir"}


def split_geez(line: str) -> tp.Iterable[str]:
    """Split Amharic text into sentences."""
    line = line.replace("፡፡", "።")
    # remove "•" if there's already EOS marker before
    line = (
        line.replace("። •", "።")
        .replace("? •", "?")
        .replace("! •", "!")
        .replace(". •", ".")
    )
    for sent in re.findall(r"[^።•!?\!\?\.]+[።•!?।৷\?\!\.]?", line, flags=re.U):
        yield sent


# ----------------------------------------------
LANGS_OLCHIKI = {"san": "san"}


def split_olchiki(line: str) -> tp.Iterable[str]:
    """Split Santali text into sentences."""
    for sent in re.findall(r"[^᱾|᱿!?\!\?]+[᱾|᱿!?\?\!]?", line, flags=re.U):
        yield sent


# test sentence: ᱱᱤᱭᱟᱹ ᱣᱤᱠᱤᱯᱤᱰᱤᱭᱟ ᱫᱚ ᱥᱟᱱᱛᱟᱲᱤ ᱛᱮ ᱚᱞ ᱟᱠᱟᱱᱟ᱾ ᱚᱨᱦᱚᱸ ᱮᱴᱟᱜ ᱯᱟᱹᱨᱥᱤᱛᱮ ᱦᱚᱸ ᱟᱭᱢᱟ ᱣᱤᱠᱤᱯᱤᱰᱤᱭᱟ ᱢᱮᱱᱟᱜᱼᱟ ᱾ ᱱᱚᱸᱰᱮ ᱠᱤᱪᱷᱩ ᱛᱟᱹᱞᱠᱟᱹ ᱮᱢ ᱦᱩᱭᱱᱟ ᱾
# splits three times


# ----------------------------------------------
LANGS_BURMESE = {"mya": "mya", "shn": "shn"}


def split_burmese(line: str) -> tp.Iterable[str]:
    """Split Amharic text into sentences."""
    # remove "•" if there's already EOS marker before
    line = line.replace("။”", "APOS။")
    for sent in re.findall(r"[^။!?\!\?\.]+[။!?।৷\?\!\.]?", line, flags=re.U):
        yield sent.replace("APOS။", "။”")


# ----------------------------------


def get_split_algo(lang: str, split_algo: str) -> tp.Callable[[str], tp.Iterable[str]]:
    # get default algorithm if requested
    if split_algo == "default":
        # use best algorithm in function of language
        if lang in LANGS_MOSES:
            split_algo = "moses"
        elif lang in LANGS_INDIC:
            split_algo = "indic"
        elif lang in LANGS_GEEZ:
            split_algo = "geez"
        elif lang in LANGS_KHMER:
            split_algo = "khmer"
        elif lang in LANGS_BURMESE:
            split_algo = "burmese"
        else:
            # use Moses by default (which likely will fall-back to English)
            split_algo = "moses"
        logger.info(f" - default algorithm for {lang} is {split_algo}")

    if split_algo == "none" or lang == "TODO":
        logger.info(" - no sentence splitting")
        return lambda line: [line]

    elif split_algo == "moses":
        if lang in LANGS_MOSES:
            lang = LANGS_MOSES[lang]
            logger.info(f" - Moses sentence splitter: using rules for '{lang}'")
        else:
            lang = "en"
            logger.info(
                f" - Moses sentence splitter for {lang}: falling back to {lang} rules"
            )
        splitter = SentenceSplitter(language=lang)
        # non_breaking_prefix_file=non_breaking_prefix_file
        return splitter.split

    elif split_algo == "indic":
        # initialize toolkit (apparently not needed for sentence segmentation)
        if INDIC_NLP_RESOURCES:
            logger.info(" - Initialize Indic NLP toolkit")
            indic_common.set_resources_path(INDIC_NLP_RESOURCES)
            indic_loader.load()
        if lang in LANGS_INDIC:
            lang = LANGS_INDIC[lang]
            logger.info(f" - Indic sentence splitter: using rules for '{lang}'")
        else:
            lang = "hi"
            logger.info(
                f" - Indic sentence splitter for {lang}: falling back to {lang} rules"
            )

        # setup normalizer
        factory = IndicNormalizerFactory()
        indic_normalizer = factory.get_normalizer(lang)

        def split_indic(line: str) -> tp.Iterable[str]:
            """Split Indian text into sentences using Indic NLP tool."""
            line = indic_normalizer.normalize(line)
            for sent in indic_sent_tok.sentence_split(line, lang=lang):
                yield sent

        return split_indic

    elif split_algo == "laonlp":
        logger.info(f" - LaoNLP sentence splitter applied to '{lang}'")
        return lao_sent_tok

    elif split_algo == "khmer":
        logger.info(f" - Khmer NLTK sentence splitter applied to '{lang}'")
        return khm_sent_tok

    elif split_algo == "bodnlp":
        logger.info(f" - Tibetan NLTK sentence splitter applied to '{lang}'")
        return bod_sent_tok

    elif split_algo == "geez":
        logger.info(f" - Ge'ez rule-based sentence splitter applied to '{lang}'")
        return split_geez

    elif split_algo == "burmese":
        logger.info(f" - Burmese rule-based sentence splitter applied to '{lang}'")
        return split_burmese

    else:
        logger.error(f"Unknown splitting algorithm {split_algo}")

    return None

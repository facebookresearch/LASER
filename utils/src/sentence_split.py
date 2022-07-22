# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
from pathlib import Path

from botok.tokenizers import sentencetokenizer as bod_sent_tok
# Indicp NLP
from indicnlp import common as indic_common
from indicnlp import loader as indic_loader
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import sentence_tokenize as indic_sent_tok
from khmernltk import sentence_tokenize as khm_sent_tok
# pythainlp for Thai
# Seahorse for Indonesian, Thai, Vietnamese
# botok for tibetan
# Spacy for
# various tool-kits
from laonlp.tokenize import sent_tokenize as lao_sent_tok
# --- sentence splitters
# Moses-style
from sentence_splitter import SentenceSplitter

INDIC_NLP_RESOURCES = None  # apparently not needed for splitting and normalization


logger = logging.getLogger("sentence_split")


split_lang_code_map = {
    "ace_Arab" : "ace_Arab",
    "ace_Latn" : "ace_Latn",
    "acm_Arab" : "acm",
    "acq_Arab" : "acq",
    "aeb_Arab" : "aeb",
    "afr_Latn" : "afr",
    "ajp_Arab" : "ajp",
    "aka_Latn" : "aka",
    "amh_Ethi" : "amh",
    "apc_Arab" : "apc",
    "arb_Arab" : "ara",
    "arb_Arab" : "ara_Arab",
    "arb_Latn" : "ara_Latn",
    "ars_Arab" : "ars",
    "ary_Arab" : "ary",
    "arz_Arab" : "arz",
    "asm_Beng" : "asm",
    "ast_Latn" : "ast",
    "awa_Deva" : "awa",
    "ayr_Latn" : "ayr",
    "azb_Arab" : "azb",
    "azj_Latn" : "azj",
    "bak_Cyrl" : "bak",
    "bam_Latn" : "bam",
    "ban_Latn" : "ban",
    "bel_Cyrl" : "bel",
    "bem_Latn" : "bem",
    "ben_Beng" : "ben",
    "bho_Deva" : "bho",
    "bjn_Arab" : "bjn_Arab",
    "bjn_Latn" : "bjn_Latn",
    "bod_Tibt" : "bod",
    "bos_Latn" : "bos",
    "bug_Latn" : "bug",
    "bul_Cyrl" : "bul",
    "cat_Latn" : "cat",
    "ceb_Latn" : "ceb",
    "ces_Latn" : "ces",
    "cjk_Latn" : "cjk",
    "ckb_Arab" : "ckb",
    "crh_Latn" : "crh_Latn",
    "cym_Latn" : "cym",
    "dan_Latn" : "dan",
    "deu_Latn" : "deu",
    "dik_Latn" : "dik",
    "diq_Latn" : "diq",
    "dyu_Latn" : "dyu",
    "dzo_Tibt" : "dzo",
    "ell_Grek" : "ell",
    "eng_Latn" : "eng",
    "epo_Latn" : "epo",
    "est_Latn" : "est",
    "eus_Latn" : "eus",
    "ewe_Latn" : "ewe",
    "fao_Latn" : "fao",
    "pes_Arab" : "fas",
    "fij_Latn" : "fij",
    "fin_Latn" : "fin",
    "fon_Latn" : "fon",
    "fra_Latn" : "fra",
    "fur_Latn" : "fur",
    "fuv_Latn" : "fuv",
    "gla_Latn" : "gla",
    "gle_Latn" : "gle",
    "glg_Latn" : "glg",
    "grn_Latn" : "grn",
    "guj_Gujr" : "guj",
    "hat_Latn" : "hat",
    "hau_Latn" : "hau",
    "heb_Hebr" : "heb",
    "hin_Deva" : "hin",
    "hne_Deva" : "hne",
    "hrv_Latn" : "hrv",
    "hun_Latn" : "hun",
    "hye_Armn" : "hye",
    "ibo_Latn" : "ibo",
    "ilo_Latn" : "ilo",
    "ind_Latn" : "ind",
    "isl_Latn" : "isl",
    "ita_Latn" : "ita",
    "jav_Latn" : "jav",
    "jpn_Jpan" : "jpn",
    "kab_Latn" : "kab",
    "kac_Latn" : "kac",
    "kam_Latn" : "kam",
    "kan_Knda" : "kan",
    "kas_Arab" : "kas_Arab",
    "kas_Deva" : "kas_Deva",
    "kat_Geor" : "kat",
    "knc_Arab" : "kau_Arab",
    "knc_Latn" : "kau_Latn",
    "kaz_Cyrl" : "kaz",
    "kbp_Latn" : "kbp",
    "kea_Latn" : "kea",
    "khm_Khmr" : "khm",
    "kik_Latn" : "kik",
    "kin_Latn" : "kin",
    "kir_Cyrl" : "kir",
    "kmb_Latn" : "kmb",
    "kon_Latn" : "kon",
    "kor_Hang" : "kor",
    "kmr_Latn" : "kur",
    "lao_Laoo" : "lao",
    "lvs_Latn" : "lav",
    "lij_Latn" : "lij",
    "lim_Latn" : "lim",
    "lin_Latn" : "lin",
    "lit_Latn" : "lit",
    "lmo_Latn" : "lmo",
    "ltg_Latn" : "ltg",
    "ltz_Latn" : "ltz",
    "lua_Latn" : "lua",
    "lug_Latn" : "lug",
    "luo_Latn" : "luo",
    "lus_Latn" : "lus",
    "mag_Deva" : "mag",
    "mai_Deva" : "mai",
    "mal_Mlym" : "mal",
    "mar_Deva" : "mar",
    "min_Arab" : "min_Arab",
    "min_Latn" : "min_Latn",
    "mkd_Cyrl" : "mkd",
    "plt_Latn" : "mlg",
    "mlt_Latn" : "mlt",
    "khk_Cyrl" : "mon",
    "mos_Latn" : "mos",
    "mri_Latn" : "mri",
    "zsm_Latn" : "msa",
    "mya_Mymr" : "mya",
    "nld_Latn" : "nld",
    "nno_Latn" : "nno",
    "nob_Latn" : "nob",
    "npi_Deva" : "npi",
    "nso_Latn" : "nso",
    "nus_Latn" : "nus",
    "nya_Latn" : "nya",
    "oci_Latn" : "oci",
    "gaz_Latn" : "orm",
    "ory_Orya" : "ory",
    "pag_Latn" : "pag",
    "pan_Guru" : "pan",
    "pap_Latn" : "pap",
    "pol_Latn" : "pol",
    "por_Latn" : "por",
    "prs_Arab" : "prs",
    "pbt_Arab" : "pus",
    "quy_Latn" : "que",
    "ron_Latn" : "ron",
    "run_Latn" : "run",
    "rus_Cyrl" : "rus",
    "sag_Latn" : "sag",
    "san_Deva" : "san",
    "sat_Olck" : "sat",
    "scn_Latn" : "scn",
    "shn_Mymr" : "shn",
    "sin_Sinh" : "sin",
    "slk_Latn" : "slk",
    "slv_Latn" : "slv",
    "smo_Latn" : "smo",
    "sna_Latn" : "sna",
    "snd_Arab" : "snd",
    "som_Latn" : "som",
    "sot_Latn" : "sot",
    "spa_Latn" : "spa",
    "als_Latn" : "sqi",
    "srd_Latn" : "srd",
    "srp_Cyrl" : "srp_Cyrl",
    "ssw_Latn" : "ssw",
    "sun_Latn" : "sun",
    "swe_Latn" : "swe",
    "swh_Latn" : "swh",
    "szl_Latn" : "szl",
    "tam_Taml" : "tam",
    "tat_Cyrl" : "tat_Cyrl",
    "tel_Telu" : "tel",
    "tgk_Cyrl" : "tgk",
    "tgl_Latn" : "tgl",
    "tha_Thai" : "tha",
    "tir_Ethi" : "tir",
    "taq_Latn" : "tmh_Latn",
    "taq_Tfng" : "tmh_Tfng",
    "ton_Latn" : "ton",
    "tpi_Latn" : "tpi",
    "tsn_Latn" : "tsn",
    "tso_Latn" : "tso",
    "tuk_Latn" : "tuk",
    "tum_Latn" : "tum",
    "tur_Latn" : "tur",
    "twi_Latn" : "twi",
    "tzm_Tfng" : "tzm",
    "uig_Arab" : "uig",
    "ukr_Cyrl" : "ukr",
    "umb_Latn" : "umb",
    "urd_Arab" : "urd",
    "uzn_Latn" : "uzb",
    "vec_Latn" : "vec",
    "vie_Latn" : "vie",
    "war_Latn" : "war",
    "wol_Latn" : "wol",
    "xho_Latn" : "xho",
    "ydd_Hebr" : "yid",
    "yor_Latn" : "yor",
    "yue_Hant" : "yue",
    "zho_Hans" : "zho_Hans",
    "zho_Hant" : "zho_Hant",
    "zul_Latn" : "zul"
}


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
    "awa": "hi",
    "ben": "bn",
    "bho": "hi",
    "brx": "bD",
    "gom": "xx",
    "guj": "gu",
    "hin": "hi",
    "hne": "hi",
    "kan": "kn",
    "kas": "hi",
    "kas_Deva": "hi",
    "kok": "kK",
    "mni": "bn",  # our meitei is in bengali script, so swapped it to bengali here
    "mag": "hi",
    "mai": "hi",
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
    if lang in split_lang_code_map:
        lang = split_lang_code_map[lang]

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

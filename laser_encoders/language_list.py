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
# -------------------------------------------------------
# Language mapping to handle different language codes and names


LASER3_LANGUAGE = {
    # akan
    "akan": "aka_Latn",
    "aka": "aka_Latn",
    "aka_Latn": "aka_Latn",
    # amharic
    "amharic": "amh_Ethi",
    "amh": "amh_Ethi",
    "amh_Ethi": "amh_Ethi",
    # assamese
    "assamese": "asm_Beng",
    "asm": "asm_Beng",
    "asm_Beng": "asm_Beng",
    # crimean tatar
    "crimean tatar": "crh_Latn",
    "crh": "crh_Latn",
    "crh_Latn": "crh_Latn",
    # chokwe
    "chokwe": "cjk_Latn",
    "cjk": "cjk_Latn",
    "cjk_Latn": "cjk_Latn",
}

LASER2_LANGUAGE = {
    "acehnese": ("ace_Latn", "ace_Arab"),
    "ace": ("ace_Latn", "ace_Arab"),
    "ace_Latn": "ace_Latn",
    "ace_Arab": "ace_Arab",
    # mesopotamian
    "mesopotamian arabic": "acm_Arab",
    "acm": "acm_Arab",
    "acm_Arab": "acm_Arab",
    # ta’izzi-adeni
    "ta’izzi-adeni arabic": "acq_Arab",
    "acq": "acq_Arab",
    "acq_Arab": "acq_Arab",
    # tunisian
    "tunisian arabic": "aeb_Arab",
    "aeb": "aeb_Arab",
    "aeb_Arab": "aeb_Arab",
    # afrikaans
    "afrikaans": "afr_Latn",
    "afr": "afr_Latn",
    "afr_Latn": "afr_Latn",
    # south levantine arabic
    "south levantine arabic": "ajp_Arab",
    "ajp": "ajp_Arab",
    "ajp_Arab": "ajp_Arab",
    # akan
    "akan": "aka_Latn",
    "aka": "aka_Latn",
    "aka_Latn": "aka_Latn",
    # amharic
    "amharic": "amh_Ethi",
    "amh": "amh_Ethi",
    "amh_Ethi": "amh_Ethi",
    # north levantine arabic
    "north levantine arabic": "apc_Arab",
    "apc": "apc_Arab",
    "apc_Arab": "apc_Arab",
    # modern standard arabic
    "modern standard arabic": ("arb_Latn", "arb_Arab"),
    "arb": ("arb_Latn", "arb_Arab"),
    "arb_Arab": "arb_Arab",
    "arb_Latn": "arb_Latn",
    # najdi arabic
    "najdi arabic": "ars_Arab",
    "ars": "ars_Arab",
    "ars_Arab": "ars_Arab",
    # moroccan arabic
    "moroccan arabic": "ary_Arab",
    "ary": "ary_Arab",
    "ary_Arab": "ary_Arab",
    # egyptian arabic
    "egyptian arabic": "arz_Arab",
    "arz": "arz_Arab",
    "arz_Arab": "arz_Arab",
    # assamese
    "assamese": "asm_Beng",
    "asm": "asm_Beng",
    "asm_Beng": "asm_Beng",
    # asturian
    "asturian": "ast_Latn",
    "ast": "ast_Latn",
    "ast_Latn": "ast_Latn",
    # awadhi
    "awadhi": "awa_Deva",
    "awa": "awa_Deva",
    "awa_Deva": "awa_Deva",
    # central aymara
    "central aymara": "ayr_Latn",
    "ayr": "ayr_Latn",
    "ayr_Latn": "ayr_Latn",
    # south azerbaijani
    "south azerbaijani": "azb_Arab",
    "azb": "azb_Arab",
    "azb_Arab": "azb_Arab",
    # north azerbaijani
    "north azerbaijani": "azj_Latn",
    "azj": "azj_Latn",
    "azj_Latn": "azj_Latn",
    # bashkir
    "bashkir": "bak_Cyrl",
    "bak": "bak_Cyrl",
    "bak_Cyrl": "bak_Cyrl",
    # bambara
    "bambara": "bam_Latn",
    "bam": "bam_Latn",
    "bam_Latn": "bam_Latn",
    # balinese
    "balinese": "ban_Latn",
    "ban": "ban_Latn",
    "ban_Latn": "ban_Latn",
    # belarusian
    "belarusian": "bel_Cyrl",
    "bel": "bel_Cyrl",
    "bel_Cyrl": "bel_Cyrl",
    # bemba
    "bemba": "bem_Latn",
    "bem": "bem_Latn",
    "bem_Latn": "bem_Latn",
    # bengali
    "bengali": "ben_Beng",
    "ben": "ben_Beng",
    "ben_Beng": "ben_Beng",
    # bhojpuri
    "bhojpuri": "bho_Deva",
    "bho": "bho_Deva",
    "bho_Deva": "bho_Deva",
    # banjar
    "banjar": ("bjn_Latn", "bjn_Arab"),
    "bjn": ("bjn_Latn", "bjn_Arab"),
    "bjn_Latn": "bjn_Latn",
    "bjn_Arab": "bjn_Arab",
    # standard tibetan
    "standard tibetan": "bod_Tibt",
    "bod": "bod_Tibt",
    "bod_Tibt": "bod_Tibt",
    # bosnian
    "bosnian": "bos_Latn",
    "bos": "bos_Latn",
    "bos_Latn": "bos_Latn",
    # buginese
    "buginese": "bug_Latn",
    "bug": "bug_Latn",
    "bug_Latn": "bug_Latn",
    # bulgarian
    "bulgarian": "bul_Cyrl",
    "bul": "bul_Cyrl",
    "bul_Cyrl": "bul_Cyrl",
    # catalan
    "catalan": "cat_Latn",
    "cat": "cat_Latn",
    "cat_Latn": "cat_Latn",
    # cebuano
    "cebuano": "ceb_Latn",
    "ceb": "ceb_Latn",
    "ceb_Latn": "ceb_Latn",
    # czech
    "czech": "ces_Latn",
    "ces": "ces_Latn",
    "ces_Latn": "ces_Latn",
    # chokwe
    "chokwe": "cjk_Latn",
    "cjk": "cjk_Latn",
    "cjk_Latn": "cjk_Latn",
    # central kurdish
    "central kurdish": "ckb_Arab",
    "ckb": "ckb_Arab",
    "ckb_Arab": "ckb_Arab",
    # crimean tatar
    "crimean tatar": "crh_Latn",
    "crh": "crh_Latn",
    "crh_Latn": "crh_Latn",
}

# langs = ["aka_Latn", "als_Latn", "amh_Ethi", "asm_Beng", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
#         "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Latn", "bod_Tibt", "bug_Latn", "ceb_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dik_Latn", "diq_Latn",
#         "dyu_Latn", "dzo_Tibt", "ewe_Latn", "fao_Latn", "fij_Latn", "fon_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "grn_Latn", "guj_Gujr",
#         "hat_Latn", "hau_Latn", "hin_Deva", "hne_Deva", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "jav_Latn", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda",
#         "kas_Arab", "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn",
#         "knc_Arab", "knc_Latn", "kon_Latn", "lao_Laoo", "lij_Latn", "lim_Latn", "lin_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn",
#         "lus_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "npi_Deva", "nso_Latn",
#         "nus_Latn", "nya_Latn", "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "prs_Arab", "quy_Latn", "run_Latn", "sag_Latn",
#         "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", "sin_Sinh", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "srd_Latn", "ssw_Latn", "sun_Latn",
#         "swh_Latn", "szl_Latn", "tam_Taml", "taq_Latn", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn",
#         "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", "war_Latn", "wol_Latn", "xho_Latn",
#         "ydd_Hebr", "yor_Latn", "zsm_Latn", "zul_Latn"]


SPM_LANGUAGE = [
    "amh_Ethi",
    "ayr_Latn",
    "azj_Latn",
    "bak_Cyrl",
    "bel_Cyrl",
    "bod_Tibt",
    "ckb_Arab",
    "crh_Latn",
    "dik_Latn",
    "dzo_Tibt",
    "fur_Latn",
    "fuv_Latn",
    "grn_Latn",
    "kab_Latn",
    "kac_Latn",
    "kaz_Cyrl",
    "kir_Cyrl",
    "kmr_Latn",
    "lij_Latn",
    "lim_Latn",
    "lmo_Latn",
    "ltg_Latn",
    "mya_Mymr",
    "pbt_Arab",
    "pes_Arab",
    "prs_Arab",
    "sat_Beng",
    "scn_Latn",
    "srd_Latn",
    "szl_Latn",
    "taq_Latn",
    "tgk_Cyrl",
    "tir_Ethi",
    "tzm_Tfng",
    "vec_Latn",
]

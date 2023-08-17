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

##################################
###### LASER 3 ###################
##################################

LASER3_LANGUAGE = {
    # acehnese
    "acehnese": "ace_Latn",
    "ace": "ace_Latn",
    "ace_Latn": "ace_Latn",
    # akan
    "akan": "aka_Latn",
    "aka": "aka_Latn",
    "aka_Latn": "aka_Latn",
    # tosk albanian
    "tosk albanian": "als_Latn",
    "als": "als_Latn",
    "als_Latn": "als_Latn",
    # amharic
    "amharic": "amh_Ethi",
    "amh": "amh_Ethi",
    "amh_Ethi": "amh_Ethi",
    # assamese
    "assamese": "asm_Beng",
    "asm": "asm_Beng",
    "asm_Beng": "asm_Beng",
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
    "banjar": "bjn_Latn",
    "bjn": "bjn_Latn",
    "bjn_Latn": "bjn_Latn",
    # standard tibetan
    "standard tibetan": "bod_Tibt",
    "bod": "bod_Tibt",
    "bod_Tibt": "bod_Tibt",
    # buginese
    "buginese": "bug_Latn",
    "bug": "bug_Latn",
    "bug_Latn": "bug_Latn",
    # cebuano
    "cebuano": "ceb_Latn",
    "ceb": "ceb_Latn",
    "ceb_Latn": "ceb_Latn",
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
    # welsh
    "welsh": "cym_Latn",
    "cym": "cym_Latn",
    "cym_Latn": "cym_Latn",
    # southwestern dinka
    "southwestern dinka": "dik_Latn",
    "dik": "dik_Latn",
    "dik_Latn": "dik_Latn",
    # southern zaza
    "southern zaza": "diq_Latn",
    "diq": "diq_Latn",
    "diq_Latn": "diq_Latn",
    # dyula
    "dyula": "dyu_Latn",
    "dyu": "dyu_Latn",
    "dyu_Latn": "dyu_Latn",
    # dzongkha
    "dzongkha": "dzo_Tibt",
    "dzo": "dzo_Tibt",
    "dzo_Tibt": "dzo_Tibt",
    # ewe
    "ewe": "ewe_Latn",
    "ewe_Latn": "ewe_Latn",
    # faroese
    "faroese": "fao_Latn",
    "fao": "fao_Latn",
    "fao_Latn": "fao_Latn",
    # fijian
    "fijian": "fij_Latn",
    "fij": "fij_Latn",
    "fij_Latn": "fij_Latn",
    # fon
    "fon": "fon_Latn",
    "fon_Latn": "fon_Latn",
    # friulian
    "friulian": "fur_Latn",
    "fur": "fur_Latn",
    "fur_Latn": "fur_Latn",
    # nigerian fulfulde
    "nigerian fulfulde": "fuv_Latn",
    "fuv": "fuv_Latn",
    "fuv_Latn": "fuv_Latn",
    # west central oromo
    "west central oromo": "gaz_Latn",
    "gaz": "gaz_Latn",
    "gaz_Latn": "gaz_Latn",
    # scottish gaelic
    "scottish gaelic": "gla_Latn",
    "gla": "gla_Latn",
    "gla_Latn": "gla_Latn",
    # irish
    "irish": "gle_Latn",
    "gle": "gle_Latn",
    "gle_Latn": "gle_Latn",
    # guarani
    "guarani": "grn_Latn",
    "grn": "grn_Latn",
    "grn_Latn": "grn_Latn",
    # gujarati
    "gujarati": "guj_Gujr",
    "guj": "guj_Gujr",
    "guj_Gujr": "guj_Gujr",
    # haitian creole
    "haitian creole": "hat_Latn",
    "hat": "hat_Latn",
    "hat_Latn": "hat_Latn",
    # hausa
    "hausa": "hau_Latn",
    "hau": "hau_Latn",
    "hau_Latn": "hau_Latn",
    # hindi
    "hindi": "hin_Deva",
    "hin": "hin_Deva",
    "hin_Deva": "hin_Deva",
    # chhattisgarhi
    "chhattisgarhi": "hne_Deva",
    "hne": "hne_Deva",
    "hne_Deva": "hne_Deva",
    # armenian
    "armenian": "hye_Armn",
    "hye": "hye_Armn",
    "hye_Armn": "hye_Armn",
    # igbo
    "igbo": "ibo_Latn",
    "ibo": "ibo_Latn",
    "ibo_Latn": "ibo_Latn",
    # ilocano
    "ilocano": "ilo_Latn",
    "ilo": "ilo_Latn",
    "ilo_Latn": "ilo_Latn",
    # indonesian
    "indonesian": "ind_Latn",
    "ind": "ind_Latn",
    "ind_Latn": "ind_Latn",
    # javanese
    "javanese": "jav_Latn",
    "jav": "jav_Latn",
    "jav_Latn": "jav_Latn",
    # kabyle
    "kabyle": "kab_Latn",
    "kab": "kab_Latn",
    "kab_Latn": "kab_Latn",
    # jingpho
    "jingpho": "kac_Latn",
    "kac": "kac_Latn",
    "kac_Latn": "kac_Latn",
    # kamba
    "kamba": "kam_Latn",
    "kam": "kam_Latn",
    "kam_Latn": "kam_Latn",
    # kannada
    "kannada": "kan_Knda",
    "kan": "kan_Knda",
    "kan_Knda": "kan_Knda",
    # kashmiri
    "kashmiri": ["kas_Arab", "kas_Deva"],
    "kas": ["kas_Arab", "kas_Deva"],
    "kas_Arab": "kas_Arab",
    "kas_Deva": "kas_Deva",
    # georgian
    "georgian": "kat_Geor",
    "kat": "kat_Geor",
    "kat_Geor": "kat_Geor",
    # kazakh
    "kazakh": "kaz_Cyrl",
    "kaz": "kaz_Cyrl",
    "kaz_Cyrl": "kaz_Cyrl",
    # kabiyè
    "kabiyè": "kbp_Latn",
    "kbp": "kbp_Latn",
    "kbp_Latn": "kbp_Latn",
    # kabuverdianu
    "kabuverdianu": "kea_Latn",
    "kea": "kea_Latn",
    "kea_Latn": "kea_Latn",
    # halh mongolian
    "halh mongolian": "khk_Cyrl",
    "khk": "khk_Cyrl",
    "khk_Cyrl": "khk_Cyrl",
    # khmer
    "khmer": "khm_Khmr",
    "khm": "khm_Khmr",
    "khm_Khmr": "khm_Khmr",
    # kikuyu
    "kikuyu": "kik_Latn",
    "kik": "kik_Latn",
    "kik_Latn": "kik_Latn",
    # kinyarwanda
    "kinyarwanda": "kin_Latn",
    "kin": "kin_Latn",
    "kin_Latn": "kin_Latn",
    # kyrgyz
    "kyrgyz": "kir_Cyrl",
    "kir": "kir_Cyrl",
    "kir_Cyrl": "kir_Cyrl",
    # kimbundu
    "kimbundu": "kmb_Latn",
    "kmb": "kmb_Latn",
    "kmb_Latn": "kmb_Latn",
    # northern kurdish
    "northern kurdish": "kmr_Latn",
    "kmr": "kmr_Latn",
    "kmr_Latn": "kmr_Latn",
    # central kanuri
    "central kanuri": ["knc_Arab", "knc_Latn"],
    "knc": ["knc_Arab", "knc_Latn"],
    "knc_Arab": "knc_Arab",
    "knc_Latn": "knc_Latn",
    # kikongo
    "kikongo": "kon_Latn",
    "kon": "kon_Latn",
    "kon_Latn": "kon_Latn",
    # lao
    "lao": "lao_Laoo",
    "lao_Laoo": "lao_Laoo",
    # ligurian
    "ligurian": "lij_Latn",
    "lij": "lij_Latn",
    "lij_Latn": "lij_Latn",
    # limburgish
    "limburgish": "lim_Latn",
    "lim": "lim_Latn",
    "lim_Latn": "lim_Latn",
    # lingala
    "lingala": "lin_Latn",
    "lin": "lin_Latn",
    "lin_Latn": "lin_Latn",
    # lombard
    "lombard": "lmo_Latn",
    "lmo": "lmo_Latn",
    "lmo_Latn": "lmo_Latn",
    # latgalian
    "latgalian": "ltg_Latn",
    "ltg": "ltg_Latn",
    "ltg_Latn": "ltg_Latn",
    # luxembourgish
    "luxembourgish": "ltz_Latn",
    "ltz": "ltz_Latn",
    "ltz_Latn": "ltz_Latn",
    # luba-kasai
    "luba-kasai": "lua_Latn",
    "lua": "lua_Latn",
    "lua_Latn": "lua_Latn",
    # ganda
    "ganda": "lug_Latn",
    "lug": "lug_Latn",
    "lug_Latn": "lug_Latn",
    # luo
    "luo": "luo_Latn",
    "luo_Latn": "luo_Latn",
    # mizo
    "mizo": "lus_Latn",
    "lus": "lus_Latn",
    "lus_Latn": "lus_Latn",
    # magahi
    "magahi": "mag_Deva",
    "mag": "mag_Deva",
    "mag_Deva": "mag_Deva",
    # maithili
    "maithili": "mai_Deva",
    "mai": "mai_Deva",
    "mai_Deva": "mai_Deva",
    # malayalam
    "malayalam": "mal_Mlym",
    "mal": "mal_Mlym",
    "mal_Mlym": "mal_Mlym",
    # marathi
    "marathi": "mar_Deva",
    "mar": "mar_Deva",
    "mar_Deva": "mar_Deva",
    # minangkabau
    "minangkabau": "min_Latn",
    "min": "min_Latn",
    "min_Latn": "min_Latn",
    # maltese
    "maltese": "mlt_Latn",
    "mlt": "mlt_Latn",
    "mlt_Latn": "mlt_Latn",
    # meitei
    "meitei": "mni_Beng",
    "mni": "mni_Beng",
    "mni_Beng": "mni_Beng",
    # mossi
    "mossi": "mos_Latn",
    "mos": "mos_Latn",
    "mos_Latn": "mos_Latn",
    # maori
    "maori": "mri_Latn",
    "mri": "mri_Latn",
    "mri_Latn": "mri_Latn",
    # burmese
    "burmese": "mya_Mymr",
    "mya": "mya_Mymr",
    "mya_Mymr": "mya_Mymr",
    # nepali
    "nepali": "npi_Deva",
    "npi": "npi_Deva",
    "npi_Deva": "npi_Deva",
    # northern sotho
    "northern sotho": "nso_Latn",
    "nso": "nso_Latn",
    "nso_Latn": "nso_Latn",
    # nuer
    "nuer": "nus_Latn",
    "nus": "nus_Latn",
    "nus_Latn": "nus_Latn",
    # nyanja
    "nyanja": "nya_Latn",
    "nya": "nya_Latn",
    "nya_Latn": "nya_Latn",
    # odia
    "odia": "ory_Orya",
    "ory": "ory_Orya",
    "ory_Orya": "ory_Orya",
    # pangasinan
    "pangasinan": "pag_Latn",
    "pag": "pag_Latn",
    "pag_Latn": "pag_Latn",
    # eastern panjabi
    "eastern panjabi": "pan_Guru",
    "pan": "pan_Guru",
    "pan_Guru": "pan_Guru",
    # papiamento
    "papiamento": "pap_Latn",
    "pap": "pap_Latn",
    "pap_Latn": "pap_Latn",
    # southern pashto
    "southern pashto": "pbt_Arab",
    "pbt": "pbt_Arab",
    "pbt_Arab": "pbt_Arab",
    # western persian
    "western persian": "pes_Arab",
    "pes": "pes_Arab",
    "pes_Arab": "pes_Arab",
    # plateau malagasy
    "plateau malagasy": "plt_Latn",
    "plt": "plt_Latn",
    "plt_Latn": "plt_Latn",
    # dari
    "dari": "prs_Arab",
    "prs": "prs_Arab",
    "prs_Arab": "prs_Arab",
    # ayacucho quechua
    "ayacucho quechua": "quy_Latn",
    "quy": "quy_Latn",
    "quy_Latn": "quy_Latn",
    # rundi
    "rundi": "run_Latn",
    "run": "run_Latn",
    "run_Latn": "run_Latn",
    # sango
    "sango": "sag_Latn",
    "sag": "sag_Latn",
    "sag_Latn": "sag_Latn",
    # sanskrit
    "sanskrit": "san_Deva",
    "san": "san_Deva",
    "san_Deva": "san_Deva",
    # santali
    "santali": "sat_Beng",
    "sat": "sat_Beng",
    "sat_Beng": "sat_Beng",
    # sicilian
    "sicilian": "scn_Latn",
    "scn": "scn_Latn",
    "scn_Latn": "scn_Latn",
    # shan
    "shan": "shn_Mymr",
    "shn": "shn_Mymr",
    "shn_Mymr": "shn_Mymr",
    # sinhala
    "sinhala": "sin_Sinh",
    "sin": "sin_Sinh",
    "sin_Sinh": "sin_Sinh",
    # samoan
    "samoan": "smo_Latn",
    "smo": "smo_Latn",
    "smo_Latn": "smo_Latn",
    # shona
    "shona": "sna_Latn",
    "sna": "sna_Latn",
    "sna_Latn": "sna_Latn",
    # sindhi
    "sindhi": "snd_Arab",
    "snd": "snd_Arab",
    "snd_Arab": "snd_Arab",
    # somali
    "somali": "som_Latn",
    "som": "som_Latn",
    "som_Latn": "som_Latn",
    # southern sotho
    "southern sotho": "sot_Latn",
    "sot": "sot_Latn",
    "sot_Latn": "sot_Latn",
    # sardinian
    "sardinian": "srd_Latn",
    "srd": "srd_Latn",
    "srd_Latn": "srd_Latn",
    # swati
    "swati": "ssw_Latn",
    "ssw": "ssw_Latn",
    "ssw_Latn": "ssw_Latn",
    # sundanese
    "sundanese": "sun_Latn",
    "sun": "sun_Latn",
    "sun_Latn": "sun_Latn",
    # swahili
    "swahili": "swh_Latn",
    "swh": "swh_Latn",
    "swh_Latn": "swh_Latn",
    # silesian
    "silesian": "szl_Latn",
    "szl": "szl_Latn",
    "szl_Latn": "szl_Latn",
    # tamil
    "tamil": "tam_Taml",
    "tam": "tam_Taml",
    "tam_Taml": "tam_Taml",
    # tamasheq
    "tamasheq": "taq_Latn",
    "taq": "taq_Latn",
    "taq_Latn": "taq_Latn",
    # tatar
    "tatar": "tat_Cyrl",
    "tat": "tat_Cyrl",
    "tat_Cyrl": "tat_Cyrl",
    # telugu
    "telugu": "tel_Telu",
    "tel": "tel_Telu",
    "tel_Telu": "tel_Telu",
    # tajik
    "tajik": "tgk_Cyrl",
    "tgk": "tgk_Cyrl",
    "tgk_Cyrl": "tgk_Cyrl",
    # tagalog
    "tagalog": "tgl_Latn",
    "tgl": "tgl_Latn",
    "tgl_Latn": "tgl_Latn",
    # thai
    "thai": "tha_Thai",
    "tha": "tha_Thai",
    "tha_Thai": "tha_Thai",
    # tigrinya
    "tigrinya": "tir_Ethi",
    "tir": "tir_Ethi",
    "tir_Ethi": "tir_Ethi",
    # tok pisin
    "tok pisin": "tpi_Latn",
    "tpi": "tpi_Latn",
    "tpi_Latn": "tpi_Latn",
    # tswana
    "tswana": "tsn_Latn",
    "tsn": "tsn_Latn",
    "tsn_Latn": "tsn_Latn",
    # tsonga
    "tsonga": "tso_Latn",
    "tso": "tso_Latn",
    "tso_Latn": "tso_Latn",
    # turkmen
    "turkmen": "tuk_Latn",
    "tuk": "tuk_Latn",
    "tuk_Latn": "tuk_Latn",
    # tumbuka
    "tumbuka": "tum_Latn",
    "tum": "tum_Latn",
    "tum_Latn": "tum_Latn",
    # turkish
    "turkish": "tur_Latn",
    "tur": "tur_Latn",
    "tur_Latn": "tur_Latn",
    # twi
    "twi": "twi_Latn",
    "twi_Latn": "twi_Latn",
    # central atlas tamazight
    "central atlas tamazight": "tzm_Tfng",
    "tzm": "tzm_Tfng",
    "tzm_Tfng": "tzm_Tfng",
    # uyghur
    "uyghur": "uig_Arab",
    "uig": "uig_Arab",
    "uig_Arab": "uig_Arab",
    # umbundu
    "umbundu": "umb_Latn",
    "umb": "umb_Latn",
    "umb_Latn": "umb_Latn",
    # urdu
    "urdu": "urd_Arab",
    "urd": "urd_Arab",
    "urd_Arab": "urd_Arab",
    # northern uzbek
    "northern uzbek": "uzn_Latn",
    "uzn": "uzn_Latn",
    "uzn_Latn": "uzn_Latn",
    # venetian
    "venetian": "vec_Latn",
    "vec": "vec_Latn",
    "vec_Latn": "vec_Latn",
    # waray
    "waray": "war_Latn",
    "war": "war_Latn",
    "war_Latn": "war_Latn",
    # wolof
    "wolof": "wol_Latn",
    "wol": "wol_Latn",
    "wol_Latn": "wol_Latn",
    # xhosa
    "xhosa": "xho_Latn",
    "xho": "xho_Latn",
    "xho_Latn": "xho_Latn",
    # eastern yiddish
    "eastern yiddish": "ydd_Hebr",
    "ydd": "ydd_Hebr",
    "ydd_Hebr": "ydd_Hebr",
    # yoruba
    "yoruba": "yor_Latn",
    "yor": "yor_Latn",
    "yor_Latn": "yor_Latn",
    # standard malay
    "standard malay": "zsm_Latn",
    "zsm": "zsm_Latn",
    "zsm_Latn": "zsm_Latn",
    # zulu
    "zulu": "zul_Latn",
    "zul": "zul_Latn",
    "zul_Latn": "zul_Latn",
}

##################################
###### LASER 2 ###################
##################################

LASER2_LANGUAGE = {
    # acehnese
    "acehnese": ["ace_Arab", "ace_Latn"],
    "ace": ["ace_Arab", "ace_Latn"],
    "ace_Arab": "ace_Arab",
    "ace_Latn": "ace_Latn",
    # mesopotamian arabic
    "mesopotamian arabic": "acm_Arab",
    "acm": "acm_Arab",
    "acm_Arab": "acm_Arab",
    # ta’izzi-adeni arabic
    "ta’izzi-adeni arabic": "acq_Arab",
    "acq": "acq_Arab",
    "acq_Arab": "acq_Arab",
    # tunisian arabic
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
    "modern standard arabic": ["arb_Arab", "arb_Latn"],
    "arb": ["arb_Arab", "arb_Latn"],
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
    "banjar": ["bjn_Arab", "bjn_Latn"],
    "bjn": ["bjn_Arab", "bjn_Latn"],
    "bjn_Arab": "bjn_Arab",
    "bjn_Latn": "bjn_Latn",
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
    # welsh
    "welsh": "cym_Latn",
    "cym": "cym_Latn",
    "cym_Latn": "cym_Latn",
    # danish
    "danish": "dan_Latn",
    "dan": "dan_Latn",
    "dan_Latn": "dan_Latn",
    # german
    "german": "deu_Latn",
    "deu": "deu_Latn",
    "deu_Latn": "deu_Latn",
    # southwestern dinka
    "southwestern dinka": "dik_Latn",
    "dik": "dik_Latn",
    "dik_Latn": "dik_Latn",
    # dyula
    "dyula": "dyu_Latn",
    "dyu": "dyu_Latn",
    "dyu_Latn": "dyu_Latn",
    # dzongkha
    "dzongkha": "dzo_Tibt",
    "dzo": "dzo_Tibt",
    "dzo_Tibt": "dzo_Tibt",
    # greek
    "greek": "ell_Grek",
    "ell": "ell_Grek",
    "ell_Grek": "ell_Grek",
    # english
    "english": "eng_Latn",
    "eng": "eng_Latn",
    "eng_Latn": "eng_Latn",
    # esperanto
    "esperanto": "epo_Latn",
    "epo": "epo_Latn",
    "epo_Latn": "epo_Latn",
    # estonian
    "estonian": "est_Latn",
    "est": "est_Latn",
    "est_Latn": "est_Latn",
    # basque
    "basque": "eus_Latn",
    "eus": "eus_Latn",
    "eus_Latn": "eus_Latn",
    # ewe
    "ewe": "ewe_Latn",
    "ewe_Latn": "ewe_Latn",
    # faroese
    "faroese": "fao_Latn",
    "fao": "fao_Latn",
    "fao_Latn": "fao_Latn",
    # fijian
    "fijian": "fij_Latn",
    "fij": "fij_Latn",
    "fij_Latn": "fij_Latn",
    # finnish
    "finnish": "fin_Latn",
    "fin": "fin_Latn",
    "fin_Latn": "fin_Latn",
    # fon
    "fon": "fon_Latn",
    "fon_Latn": "fon_Latn",
    # french
    "french": "fra_Latn",
    "fra": "fra_Latn",
    "fra_Latn": "fra_Latn",
    # friulian
    "friulian": "fur_Latn",
    "fur": "fur_Latn",
    "fur_Latn": "fur_Latn",
    # nigerian fulfulde
    "nigerian fulfulde": "fuv_Latn",
    "fuv": "fuv_Latn",
    "fuv_Latn": "fuv_Latn",
    # scottish gaelic
    "scottish gaelic": "gla_Latn",
    "gla": "gla_Latn",
    "gla_Latn": "gla_Latn",
    # irish
    "irish": "gle_Latn",
    "gle": "gle_Latn",
    "gle_Latn": "gle_Latn",
    # galician
    "galician": "glg_Latn",
    "glg": "glg_Latn",
    "glg_Latn": "glg_Latn",
    # guarani
    "guarani": "grn_Latn",
    "grn": "grn_Latn",
    "grn_Latn": "grn_Latn",
    # gujarati
    "gujarati": "guj_Gujr",
    "guj": "guj_Gujr",
    "guj_Gujr": "guj_Gujr",
    # haitian creole
    "haitian creole": "hat_Latn",
    "hat": "hat_Latn",
    "hat_Latn": "hat_Latn",
    # hausa
    "hausa": "hau_Latn",
    "hau": "hau_Latn",
    "hau_Latn": "hau_Latn",
    # hebrew
    "hebrew": "heb_Hebr",
    "heb": "heb_Hebr",
    "heb_Hebr": "heb_Hebr",
    # hindi
    "hindi": "hin_Deva",
    "hin": "hin_Deva",
    "hin_Deva": "hin_Deva",
    # chhattisgarhi
    "chhattisgarhi": "hne_Deva",
    "hne": "hne_Deva",
    "hne_Deva": "hne_Deva",
    # croatian
    "croatian": "hrv_Latn",
    "hrv": "hrv_Latn",
    "hrv_Latn": "hrv_Latn",
    # hungarian
    "hungarian": "hun_Latn",
    "hun": "hun_Latn",
    "hun_Latn": "hun_Latn",
    # armenian
    "armenian": "hye_Armn",
    "hye": "hye_Armn",
    "hye_Armn": "hye_Armn",
    # igbo
    "igbo": "ibo_Latn",
    "ibo": "ibo_Latn",
    "ibo_Latn": "ibo_Latn",
    # ilocano
    "ilocano": "ilo_Latn",
    "ilo": "ilo_Latn",
    "ilo_Latn": "ilo_Latn",
    # indonesian
    "indonesian": "ind_Latn",
    "ind": "ind_Latn",
    "ind_Latn": "ind_Latn",
    # icelandic
    "icelandic": "isl_Latn",
    "isl": "isl_Latn",
    "isl_Latn": "isl_Latn",
    # italian
    "italian": "ita_Latn",
    "ita": "ita_Latn",
    "ita_Latn": "ita_Latn",
    # javanese
    "javanese": "jav_Latn",
    "jav": "jav_Latn",
    "jav_Latn": "jav_Latn",
    # japanese
    "japanese": "jpn_Jpan",
    "jpn": "jpn_Jpan",
    "jpn_Jpan": "jpn_Jpan",
    # kabyle
    "kabyle": "kab_Latn",
    "kab": "kab_Latn",
    "kab_Latn": "kab_Latn",
    # jingpho
    "jingpho": "kac_Latn",
    "kac": "kac_Latn",
    "kac_Latn": "kac_Latn",
    # kamba
    "kamba": "kam_Latn",
    "kam": "kam_Latn",
    "kam_Latn": "kam_Latn",
    # kannada
    "kannada": "kan_Knda",
    "kan": "kan_Knda",
    "kan_Knda": "kan_Knda",
    # kashmiri
    "kashmiri": ["kas_Arab", "kas_Deva"],
    "kas": ["kas_Arab", "kas_Deva"],
    "kas_Arab": "kas_Arab",
    "kas_Deva": "kas_Deva",
    # georgian
    "georgian": "kat_Geor",
    "kat": "kat_Geor",
    "kat_Geor": "kat_Geor",
    # central kanuri
    "central kanuri": ["knc_Arab", "knc_Latn"],
    "knc": ["knc_Arab", "knc_Latn"],
    "knc_Arab": "knc_Arab",
    "knc_Latn": "knc_Latn",
    # kazakh
    "kazakh": "kaz_Cyrl",
    "kaz": "kaz_Cyrl",
    "kaz_Cyrl": "kaz_Cyrl",
    # kabiyè
    "kabiyè": "kbp_Latn",
    "kbp": "kbp_Latn",
    "kbp_Latn": "kbp_Latn",
    # kabuverdianu
    "kabuverdianu": "kea_Latn",
    "kea": "kea_Latn",
    "kea_Latn": "kea_Latn",
    # khmer
    "khmer": "khm_Khmr",
    "khm": "khm_Khmr",
    "khm_Khmr": "khm_Khmr",
    # kikuyu
    "kikuyu": "kik_Latn",
    "kik": "kik_Latn",
    "kik_Latn": "kik_Latn",
    # kinyarwanda
    "kinyarwanda": "kin_Latn",
    "kin": "kin_Latn",
    "kin_Latn": "kin_Latn",
    # kyrgyz
    "kyrgyz": "kir_Cyrl",
    "kir": "kir_Cyrl",
    "kir_Cyrl": "kir_Cyrl",
    # kimbundu
    "kimbundu": "kmb_Latn",
    "kmb": "kmb_Latn",
    "kmb_Latn": "kmb_Latn",
    # northern kurdish
    "northern kurdish": "kmr_Latn",
    "kmr": "kmr_Latn",
    "kmr_Latn": "kmr_Latn",
    # kikongo
    "kikongo": "kon_Latn",
    "kon": "kon_Latn",
    "kon_Latn": "kon_Latn",
    # korean
    "korean": "kor_Hang",
    "kor": "kor_Hang",
    "kor_Hang": "kor_Hang",
    # lao
    "lao": "lao_Laoo",
    "lao_Laoo": "lao_Laoo",
    # ligurian
    "ligurian": "lij_Latn",
    "lij": "lij_Latn",
    "lij_Latn": "lij_Latn",
    # limburgish
    "limburgish": "lim_Latn",
    "lim": "lim_Latn",
    "lim_Latn": "lim_Latn",
    # lingala
    "lingala": "lin_Latn",
    "lin": "lin_Latn",
    "lin_Latn": "lin_Latn",
    # lithuanian
    "lithuanian": "lit_Latn",
    "lit": "lit_Latn",
    "lit_Latn": "lit_Latn",
    # lombard
    "lombard": "lmo_Latn",
    "lmo": "lmo_Latn",
    "lmo_Latn": "lmo_Latn",
    # latgalian
    "latgalian": "ltg_Latn",
    "ltg": "ltg_Latn",
    "ltg_Latn": "ltg_Latn",
    # luxembourgish
    "luxembourgish": "ltz_Latn",
    "ltz": "ltz_Latn",
    "ltz_Latn": "ltz_Latn",
    # luba-kasai
    "luba-kasai": "lua_Latn",
    "lua": "lua_Latn",
    "lua_Latn": "lua_Latn",
    # ganda
    "ganda": "lug_Latn",
    "lug": "lug_Latn",
    "lug_Latn": "lug_Latn",
    # luo
    "luo": "luo_Latn",
    "luo_Latn": "luo_Latn",
    # mizo
    "mizo": "lus_Latn",
    "lus": "lus_Latn",
    "lus_Latn": "lus_Latn",
    # standard latvian
    "standard latvian": "lvs_Latn",
    "lvs": "lvs_Latn",
    "lvs_Latn": "lvs_Latn",
    # magahi
    "magahi": "mag_Deva",
    "mag": "mag_Deva",
    "mag_Deva": "mag_Deva",
    # maithili
    "maithili": "mai_Deva",
    "mai": "mai_Deva",
    "mai_Deva": "mai_Deva",
    # malayalam
    "malayalam": "mal_Mlym",
    "mal": "mal_Mlym",
    "mal_Mlym": "mal_Mlym",
    # marathi
    "marathi": "mar_Deva",
    "mar": "mar_Deva",
    "mar_Deva": "mar_Deva",
    # minangkabau
    "minangkabau": ["min_Arab", "min_Latn"],
    "min": ["min_Arab", "min_Latn"],
    "min_Arab": "min_Arab",
    "min_Latn": "min_Latn",
    # macedonian
    "macedonian": "mkd_Cyrl",
    "mkd": "mkd_Cyrl",
    "mkd_Cyrl": "mkd_Cyrl",
    # plateau malagasy
    "plateau malagasy": "plt_Latn",
    "plt": "plt_Latn",
    "plt_Latn": "plt_Latn",
    # maltese
    "maltese": "mlt_Latn",
    "mlt": "mlt_Latn",
    "mlt_Latn": "mlt_Latn",
    # meitei
    "meitei": "mni_Beng",
    "mni": "mni_Beng",
    "mni_Beng": "mni_Beng",
    # halh mongolian
    "halh mongolian": "khk_Cyrl",
    "khk": "khk_Cyrl",
    "khk_Cyrl": "khk_Cyrl",
    # mossi
    "mossi": "mos_Latn",
    "mos": "mos_Latn",
    "mos_Latn": "mos_Latn",
    # maori
    "maori": "mri_Latn",
    "mri": "mri_Latn",
    "mri_Latn": "mri_Latn",
    # burmese
    "burmese": "mya_Mymr",
    "mya": "mya_Mymr",
    "mya_Mymr": "mya_Mymr",
    # dutch
    "dutch": "nld_Latn",
    "nld": "nld_Latn",
    "nld_Latn": "nld_Latn",
    # norwegian nynorsk
    "norwegian nynorsk": "nno_Latn",
    "nno": "nno_Latn",
    "nno_Latn": "nno_Latn",
    # norwegian bokmål
    "norwegian bokmål": "nob_Latn",
    "nob": "nob_Latn",
    "nob_Latn": "nob_Latn",
    # nepali
    "nepali": "npi_Deva",
    "npi": "npi_Deva",
    "npi_Deva": "npi_Deva",
    # northern sotho
    "northern sotho": "nso_Latn",
    "nso": "nso_Latn",
    "nso_Latn": "nso_Latn",
    # nuer
    "nuer": "nus_Latn",
    "nus": "nus_Latn",
    "nus_Latn": "nus_Latn",
    # nyanja
    "nyanja": "nya_Latn",
    "nya": "nya_Latn",
    "nya_Latn": "nya_Latn",
    # occitan
    "occitan": "oci_Latn",
    "oci": "oci_Latn",
    "oci_Latn": "oci_Latn",
    # west central oromo
    "west central oromo": "gaz_Latn",
    "gaz": "gaz_Latn",
    "gaz_Latn": "gaz_Latn",
    # odia
    "odia": "ory_Orya",
    "ory": "ory_Orya",
    "ory_Orya": "ory_Orya",
    # pangasinan
    "pangasinan": "pag_Latn",
    "pag": "pag_Latn",
    "pag_Latn": "pag_Latn",
    # eastern panjabi
    "eastern panjabi": "pan_Guru",
    "pan": "pan_Guru",
    "pan_Guru": "pan_Guru",
    # papiamento
    "papiamento": "pap_Latn",
    "pap": "pap_Latn",
    "pap_Latn": "pap_Latn",
    # western persian
    "western persian": "pes_Arab",
    "pes": "pes_Arab",
    "pes_Arab": "pes_Arab",
    # polish
    "polish": "pol_Latn",
    "pol": "pol_Latn",
    "pol_Latn": "pol_Latn",
    # portuguese
    "portuguese": "por_Latn",
    "por": "por_Latn",
    "por_Latn": "por_Latn",
    # dari
    "dari": "prs_Arab",
    "prs": "prs_Arab",
    "prs_Arab": "prs_Arab",
    # southern pashto
    "southern pashto": "pbt_Arab",
    "pbt": "pbt_Arab",
    "pbt_Arab": "pbt_Arab",
    # ayacucho quechua
    "ayacucho quechua": "quy_Latn",
    "quy": "quy_Latn",
    "quy_Latn": "quy_Latn",
    # romanian
    "romanian": "ron_Latn",
    "ron": "ron_Latn",
    "ron_Latn": "ron_Latn",
    # rundi
    "rundi": "run_Latn",
    "run": "run_Latn",
    "run_Latn": "run_Latn",
    # russian
    "russian": "rus_Cyrl",
    "rus": "rus_Cyrl",
    "rus_Cyrl": "rus_Cyrl",
    # sango
    "sango": "sag_Latn",
    "sag": "sag_Latn",
    "sag_Latn": "sag_Latn",
    # sanskrit
    "sanskrit": "san_Deva",
    "san": "san_Deva",
    "san_Deva": "san_Deva",
    # santali
    "santali": "sat_Olck",
    "sat": "sat_Olck",
    "sat_Olck": "sat_Olck",
    # sicilian
    "sicilian": "scn_Latn",
    "scn": "scn_Latn",
    "scn_Latn": "scn_Latn",
    # shan
    "shan": "shn_Mymr",
    "shn": "shn_Mymr",
    "shn_Mymr": "shn_Mymr",
    # sinhala
    "sinhala": "sin_Sinh",
    "sin": "sin_Sinh",
    "sin_Sinh": "sin_Sinh",
    # slovak
    "slovak": "slk_Latn",
    "slk": "slk_Latn",
    "slk_Latn": "slk_Latn",
    # slovenian
    "slovenian": "slv_Latn",
    "slv": "slv_Latn",
    "slv_Latn": "slv_Latn",
    # samoan
    "samoan": "smo_Latn",
    "smo": "smo_Latn",
    "smo_Latn": "smo_Latn",
    # shona
    "shona": "sna_Latn",
    "sna": "sna_Latn",
    "sna_Latn": "sna_Latn",
    # sindhi
    "sindhi": "snd_Arab",
    "snd": "snd_Arab",
    "snd_Arab": "snd_Arab",
    # somali
    "somali": "som_Latn",
    "som": "som_Latn",
    "som_Latn": "som_Latn",
    # southern sotho
    "southern sotho": "sot_Latn",
    "sot": "sot_Latn",
    "sot_Latn": "sot_Latn",
    # spanish
    "spanish": "spa_Latn",
    "spa": "spa_Latn",
    "spa_Latn": "spa_Latn",
    # tosk albanian
    "tosk albanian": "als_Latn",
    "als": "als_Latn",
    "als_Latn": "als_Latn",
    # sardinian
    "sardinian": "srd_Latn",
    "srd": "srd_Latn",
    "srd_Latn": "srd_Latn",
    # serbian
    "serbian": "srp_Cyrl",
    "srp": "srp_Cyrl",
    "srp_Cyrl": "srp_Cyrl",
    # swati
    "swati": "ssw_Latn",
    "ssw": "ssw_Latn",
    "ssw_Latn": "ssw_Latn",
    # sundanese
    "sundanese": "sun_Latn",
    "sun": "sun_Latn",
    "sun_Latn": "sun_Latn",
    # swedish
    "swedish": "swe_Latn",
    "swe": "swe_Latn",
    "swe_Latn": "swe_Latn",
    # swahili
    "swahili": "swh_Latn",
    "swh": "swh_Latn",
    "swh_Latn": "swh_Latn",
    # silesian
    "silesian": "szl_Latn",
    "szl": "szl_Latn",
    "szl_Latn": "szl_Latn",
    # tamil
    "tamil": "tam_Taml",
    "tam": "tam_Taml",
    "tam_Taml": "tam_Taml",
    # tatar
    "tatar": "tat_Cyrl",
    "tat": "tat_Cyrl",
    "tat_Cyrl": "tat_Cyrl",
    # telugu
    "telugu": "tel_Telu",
    "tel": "tel_Telu",
    "tel_Telu": "tel_Telu",
    # tajik
    "tajik": "tgk_Cyrl",
    "tgk": "tgk_Cyrl",
    "tgk_Cyrl": "tgk_Cyrl",
    # tagalog
    "tagalog": "tgl_Latn",
    "tgl": "tgl_Latn",
    "tgl_Latn": "tgl_Latn",
    # thai
    "thai": "tha_Thai",
    "tha": "tha_Thai",
    "tha_Thai": "tha_Thai",
    # tigrinya
    "tigrinya": "tir_Ethi",
    "tir": "tir_Ethi",
    "tir_Ethi": "tir_Ethi",
    # tamasheq
    "tamasheq": ["taq_Latn", "taq_Tfng"],
    "taq": ["taq_Latn", "taq_Tfng"],
    "taq_Latn": "taq_Latn",
    "taq_Tfng": "taq_Tfng",
    # tok pisin
    "tok pisin": "tpi_Latn",
    "tpi": "tpi_Latn",
    "tpi_Latn": "tpi_Latn",
    # tswana
    "tswana": "tsn_Latn",
    "tsn": "tsn_Latn",
    "tsn_Latn": "tsn_Latn",
    # tsonga
    "tsonga": "tso_Latn",
    "tso": "tso_Latn",
    "tso_Latn": "tso_Latn",
    # turkmen
    "turkmen": "tuk_Latn",
    "tuk": "tuk_Latn",
    "tuk_Latn": "tuk_Latn",
    # tumbuka
    "tumbuka": "tum_Latn",
    "tum": "tum_Latn",
    "tum_Latn": "tum_Latn",
    # turkish
    "turkish": "tur_Latn",
    "tur": "tur_Latn",
    "tur_Latn": "tur_Latn",
    # twi
    "twi": "twi_Latn",
    "twi_Latn": "twi_Latn",
    # central atlas tamazight
    "central atlas tamazight": "tzm_Tfng",
    "tzm": "tzm_Tfng",
    "tzm_Tfng": "tzm_Tfng",
    # uyghur
    "uyghur": "uig_Arab",
    "uig": "uig_Arab",
    "uig_Arab": "uig_Arab",
    # ukrainian
    "ukrainian": "ukr_Cyrl",
    "ukr": "ukr_Cyrl",
    "ukr_Cyrl": "ukr_Cyrl",
    # umbundu
    "umbundu": "umb_Latn",
    "umb": "umb_Latn",
    "umb_Latn": "umb_Latn",
    # urdu
    "urdu": "urd_Arab",
    "urd": "urd_Arab",
    "urd_Arab": "urd_Arab",
    # northern uzbek
    "northern uzbek": "uzn_Latn",
    "uzn": "uzn_Latn",
    "uzn_Latn": "uzn_Latn",
    # venetian
    "venetian": "vec_Latn",
    "vec": "vec_Latn",
    "vec_Latn": "vec_Latn",
    # vietnamese
    "vietnamese": "vie_Latn",
    "vie": "vie_Latn",
    "vie_Latn": "vie_Latn",
    # waray
    "waray": "war_Latn",
    "war": "war_Latn",
    "war_Latn": "war_Latn",
    # wolof
    "wolof": "wol_Latn",
    "wol": "wol_Latn",
    "wol_Latn": "wol_Latn",
    # xhosa
    "xhosa": "xho_Latn",
    "xho": "xho_Latn",
    "xho_Latn": "xho_Latn",
    # eastern yiddish
    "eastern yiddish": "ydd_Hebr",
    "ydd": "ydd_Hebr",
    "ydd_Hebr": "ydd_Hebr",
    # yoruba
    "yoruba": "yor_Latn",
    "yor": "yor_Latn",
    "yor_Latn": "yor_Latn",
    # yue chinese
    "yue chinese": "yue_Hant",
    "yue": "yue_Hant",
    "yue_Hant": "yue_Hant",
    # chinese
    "chinese": ["zho_Hans", "zho_Hant"],
    "zho": ["zho_Hans", "zho_Hant"],
    "zho_Hans": "zho_Hans",
    "zho_Hant": "zho_Hant",
    # standard malay
    "standard malay": "zsm_Latn",
    "zsm": "zsm_Latn",
    "zsm_Latn": "zsm_Latn",
    # zulu
    "zulu": "zul_Latn",
    "zul": "zul_Latn",
    "zul_Latn": "zul_Latn",
}

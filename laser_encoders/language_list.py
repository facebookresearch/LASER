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

def build_language_names_dict(language_list, language_names):
    result_dict = {}

    for lang_code in language_list:
        if lang_code in language_names:
            names_list = language_names[lang_code]

            # Check if the names_list is a list or a string
            if isinstance(names_list, list):
                for name in names_list:
                    if name not in result_dict:
                        result_dict[name] = []
                    result_dict[name].append(lang_code)
            else:
                # Modified this part to handle a single element without creating a list
                if names_list not in result_dict:
                    result_dict[names_list] = []
                result_dict[names_list].append(lang_code)

    # Remove single-element lists and convert them to the element itself
    for key in result_dict:
        if len(result_dict[key]) == 1:
            result_dict[key] = result_dict[key][0]

    return result_dict

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
###### LANGUAGE NAMES ############
##################################

LANGUAGE_NAMES = {
    "ace_Arab": ["acehnese", "ace", "ace_Arab"],
    "ace_Latn": ["acehnese", "ace", "ace_Latn"],
    "acm_Arab": ["mesopotamian arabic", "acm", "acm_Arab"],
    "acq_Arab": ["ta’izzi-adeni arabic", "acq", "acq_Arab"],
    "aeb_Arab": ["tunisian arabic", "aeb", "aeb_Arab"],
    "afr_Latn": ["afrikaans", "afr", "afr_Latn"],
    "ajp_Arab": ["south levantine arabic", "ajp", "ajp_Arab"],
    "aka_Latn": ["akan", "aka", "aka_Latn"],
    "amh_Ethi": ["amharic", "amh", "amh_Ethi"],
    "apc_Arab": ["north levantine arabic", "apc", "apc_Arab"],
    "arb_Arab": ["modern standard arabic", "arb", "arb_Arab"],
    "arb_Latn": ["modern standard arabic", "arb", "arb_Latn"],
    "ars_Arab": ["najdi arabic", "ars", "ars_Arab"],
    "ary_Arab": ["moroccan arabic", "ary", "ary_Arab"],
    "arz_Arab": ["egyptian arabic", "arz", "arz_Arab"],
    "asm_Beng": ["assamese", "asm", "asm_Beng"],
    "ast_Latn": ["asturian", "ast", "ast_Latn"],
    "awa_Deva": ["awadhi", "awa", "awa_Deva"],
    "ayr_Latn": ["central aymara", "ayr", "ayr_Latn"],
    "azb_Arab": ["south azerbaijani", "azb", "azb_Arab"],
    "azj_Latn": ["north azerbaijani", "azj", "azj_Latn"],
    "bak_Cyrl": ["bashkir", "bak", "bak_Cyrl"],
    "bam_Latn": ["bambara", "bam", "bam_Latn"],
    "ban_Latn": ["balinese", "ban", "ban_Latn"],
    "bel_Cyrl": ["belarusian", "bel", "bel_Cyrl"],
    "bem_Latn": ["bemba", "bem", "bem_Latn"],
    "ben_Beng": ["bengali", "ben", "ben_Beng"],
    "bho_Deva": ["bhojpuri", "bho", "bho_Deva"],
    "bjn_Arab": ["banjar", "bjn", "bjn_Arab"],
    "bjn_Latn": ["banjar", "bjn", "bjn_Latn"],
    "bod_Tibt": ["standard tibetan", "bod", "bod_Tibt"],
    "bos_Latn": ["bosnian", "bos", "bos_Latn"],
    "bug_Latn": ["buginese", "bug", "bug_Latn"],
    "bul_Cyrl": ["bulgarian", "bul", "bul_Cyrl"],
    "cat_Latn": ["catalan", "cat", "cat_Latn"],
    "ceb_Latn": ["cebuano", "ceb", "ceb_Latn"],
    "ces_Latn": ["czech", "ces", "ces_Latn"],
    "cjk_Latn": ["chokwe", "cjk", "cjk_Latn"],
    "ckb_Arab": ["central kurdish", "ckb", "ckb_Arab"],
    "crh_Latn": ["crimean tatar", "crh", "crh_Latn"],
    "cym_Latn": ["welsh", "cym", "cym_Latn"],
    "dan_Latn": ["danish", "dan", "dan_Latn"],
    "deu_Latn": ["german", "deu", "deu_Latn"],
    "dik_Latn": ["southwestern dinka", "dik", "dik_Latn"],
    "dyu_Latn": ["dyula", "dyu", "dyu_Latn"],
    "dzo_Tibt": ["dzongkha", "dzo", "dzo_Tibt"],
    "ell_Grek": ["greek", "ell", "ell_Grek"],
    "eng_Latn": ["english", "eng", "eng_Latn"],
    "epo_Latn": ["esperanto", "epo", "epo_Latn"],
    "est_Latn": ["estonian", "est", "est_Latn"],
    "eus_Latn": ["basque", "eus", "eus_Latn"],
    "ewe_Latn": ["ewe", "ewe_Latn"],
    "fao_Latn": ["faroese", "fao", "fao_Latn"],
    "fij_Latn": ["fijian", "fij", "fij_Latn"],
    "fin_Latn": ["finnish", "fin", "fin_Latn"],
    "fon_Latn": ["fon", "fon_Latn"],
    "fra_Latn": ["french", "fra", "fra_Latn"],
    "fur_Latn": ["friulian", "fur", "fur_Latn"],
    "fuv_Latn": ["nigerian fulfulde", "fuv", "fuv_Latn"],
    "gla_Latn": ["scottish gaelic", "gla", "gla_Latn"],
    "gle_Latn": ["irish", "gle", "gle_Latn"],
    "glg_Latn": ["galician", "glg", "glg_Latn"],
    "grn_Latn": ["guarani", "grn", "grn_Latn"],
    "guj_Gujr": ["gujarati", "guj", "guj_Gujr"],
    "hat_Latn": ["haitian creole", "hat", "hat_Latn"],
    "hau_Latn": ["hausa", "hau", "hau_Latn"],
    "heb_Hebr": ["hebrew", "heb", "heb_Hebr"],
    "hin_Deva": ["hindi", "hin", "hin_Deva"],
    "hne_Deva": ["chhattisgarhi", "hne", "hne_Deva"],
    "hrv_Latn": ["croatian", "hrv", "hrv_Latn"],
    "hun_Latn": ["hungarian", "hun", "hun_Latn"],
    "hye_Armn": ["armenian", "hye", "hye_Armn"],
    "ibo_Latn": ["igbo", "ibo", "ibo_Latn"],
    "ilo_Latn": ["ilocano", "ilo", "ilo_Latn"],
    "ind_Latn": ["indonesian", "ind", "ind_Latn"],
    "isl_Latn": ["icelandic", "isl", "isl_Latn"],
    "ita_Latn": ["italian", "ita", "ita_Latn"],
    "jav_Latn": ["javanese", "jav", "jav_Latn"],
    "jpn_Jpan": ["japanese", "jpn", "jpn_Jpan"],
    "kab_Latn": ["kabyle", "kab", "kab_Latn"],
    "kac_Latn": ["jingpho", "kac", "kac_Latn"],
    "kam_Latn": ["kamba", "kam", "kam_Latn"],
    "kan_Knda": ["kannada", "kan", "kan_Knda"],
    "kas_Arab": ["kashmiri", "kas", "kas_Arab"],
    "kas_Deva": ["kashmiri", "kas", "kas_Deva"],
    "kat_Geor": ["georgian", "kat", "kat_Geor"],
    "knc_Arab": ["central kanuri", "knc", "knc_Arab"],
    "knc_Latn": ["central kanuri", "knc", "knc_Latn"],
    "kaz_Cyrl": ["kazakh", "kaz", "kaz_Cyrl"],
    "kbp_Latn": ["kabiyè", "kbp", "kbp_Latn"],
    "kea_Latn": ["kabuverdianu", "kea", "kea_Latn"],
    "khm_Khmr": ["khmer", "khm", "khm_Khmr"],
    "kik_Latn": ["kikuyu", "kik", "kik_Latn"],
    "kin_Latn": ["kinyarwanda", "kin", "kin_Latn"],
    "kir_Cyrl": ["kyrgyz", "kir", "kir_Cyrl"],
    "kmb_Latn": ["kimbundu", "kmb", "kmb_Latn"],
    "kmr_Latn": ["northern kurdish", "kmr", "kmr_Latn"],
    "kon_Latn": ["kikongo", "kon", "kon_Latn"],
    "kor_Hang": ["korean", "kor", "kor_Hang"],
    "lao_Laoo": ["lao", "lao_Laoo"],
    "lij_Latn": ["ligurian", "lij", "lij_Latn"],
    "lim_Latn": ["limburgish", "lim", "lim_Latn"],
    "lin_Latn": ["lingala", "lin", "lin_Latn"],
    "lit_Latn": ["lithuanian", "lit", "lit_Latn"],
    "lmo_Latn": ["lombard", "lmo", "lmo_Latn"],
    "ltg_Latn": ["latgalian", "ltg", "ltg_Latn"],
    "ltz_Latn": ["luxembourgish", "ltz", "ltz_Latn"],
    "lua_Latn": ["luba-kasai", "lua", "lua_Latn"],
    "lug_Latn": ["ganda", "lug", "lug_Latn"],
    "luo_Latn": ["luo", "luo_Latn"],
    "lus_Latn": ["mizo", "lus", "lus_Latn"],
    "lvs_Latn": ["standard latvian", "lvs", "lvs_Latn"],
    "mag_Deva": ["magahi", "mag", "mag_Deva"],
    "mai_Deva": ["maithili", "mai", "mai_Deva"],
    "mal_Mlym": ["malayalam", "mal", "mal_Mlym"],
    "mar_Deva": ["marathi", "mar", "mar_Deva"],
    "min_Arab": ["minangkabau", "min", "min_Arab"],
    "min_Latn": ["minangkabau", "min", "min_Latn"],
    "mkd_Cyrl": ["macedonian", "mkd", "mkd_Cyrl"],
    "plt_Latn": ["plateau malagasy", "plt", "plt_Latn"],
    "mlt_Latn": ["maltese", "mlt", "mlt_Latn"],
    "mni_Beng": ["meitei", "mni", "mni_Beng"],
    "khk_Cyrl": ["halh mongolian", "khk", "khk_Cyrl"],
    "mos_Latn": ["mossi", "mos", "mos_Latn"],
    "mri_Latn": ["maori", "mri", "mri_Latn"],
    "mya_Mymr": ["burmese", "mya", "mya_Mymr"],
    "nld_Latn": ["dutch", "nld", "nld_Latn"],
    "nno_Latn": ["norwegian nynorsk", "nno", "nno_Latn"],
    "nob_Latn": ["norwegian bokmål", "nob", "nob_Latn"],
    "npi_Deva": ["nepali", "npi", "npi_Deva"],
    "nso_Latn": ["northern sotho", "nso", "nso_Latn"],
    "nus_Latn": ["nuer", "nus", "nus_Latn"],
    "nya_Latn": ["nyanja", "nya", "nya_Latn"],
    "oci_Latn": ["occitan", "oci", "oci_Latn"],
    "gaz_Latn": ["west central oromo", "gaz", "gaz_Latn"],
    "ory_Orya": ["odia", "ory", "ory_Orya"],
    "pag_Latn": ["pangasinan", "pag", "pag_Latn"],
    "pan_Guru": ["eastern panjabi", "pan", "pan_Guru"],
    "pap_Latn": ["papiamento", "pap", "pap_Latn"],
    "pes_Arab": ["western persian", "pes", "pes_Arab"],
    "pol_Latn": ["polish", "pol", "pol_Latn"],
    "por_Latn": ["portuguese", "por", "por_Latn"],
    "prs_Arab": ["dari", "prs", "prs_Arab"],
    "pbt_Arab": ["southern pashto", "pbt", "pbt_Arab"],
    "quy_Latn": ["ayacucho quechua", "quy", "quy_Latn"],
    "ron_Latn": ["romanian", "ron", "ron_Latn"],
    "run_Latn": ["rundi", "run", "run_Latn"],
    "rus_Cyrl": ["russian", "rus", "rus_Cyrl"],
    "sag_Latn": ["sango", "sag", "sag_Latn"],
    "san_Deva": ["sanskrit", "san", "san_Deva"],
    "sat_Olck": ["santali", "sat", "sat_Olck"],
    "scn_Latn": ["sicilian", "scn", "scn_Latn"],
    "shn_Mymr": ["shan", "shn", "shn_Mymr"],
    "sin_Sinh": ["sinhala", "sin", "sin_Sinh"],
    "slk_Latn": ["slovak", "slk", "slk_Latn"],
    "slv_Latn": ["slovenian", "slv", "slv_Latn"],
    "smo_Latn": ["samoan", "smo", "smo_Latn"],
    "sna_Latn": ["shona", "sna", "sna_Latn"],
    "snd_Arab": ["sindhi", "snd", "snd_Arab"],
    "som_Latn": ["somali", "som", "som_Latn"],
    "sot_Latn": ["southern sotho", "sot", "sot_Latn"],
    "spa_Latn": ["spanish", "spa", "spa_Latn"],
    "als_Latn": ["tosk albanian", "als", "als_Latn"],
    "srd_Latn": ["sardinian", "srd", "srd_Latn"],
    "srp_Cyrl": ["serbian", "srp", "srp_Cyrl"],
    "ssw_Latn": ["swati", "ssw", "ssw_Latn"],
    "sun_Latn": ["sundanese", "sun", "sun_Latn"],
    "swe_Latn": ["swedish", "swe", "swe_Latn"],
    "swh_Latn": ["swahili", "swh", "swh_Latn"],
    "szl_Latn": ["silesian", "szl", "szl_Latn"],
    "tam_Taml": ["tamil", "tam", "tam_Taml"],
    "tat_Cyrl": ["tatar", "tat", "tat_Cyrl"],
    "tel_Telu": ["telugu", "tel", "tel_Telu"],
    "tgk_Cyrl": ["tajik", "tgk", "tgk_Cyrl"],
    "tgl_Latn": ["tagalog", "tgl", "tgl_Latn"],
    "tha_Thai": ["thai", "tha", "tha_Thai"],
    "tir_Ethi": ["tigrinya", "tir", "tir_Ethi"],
    "taq_Latn": ["tamasheq", "taq", "taq_Latn"],
    "taq_Tfng": ["tamasheq", "taq", "taq_Tfng"],
    "tpi_Latn": ["tok pisin", "tpi", "tpi_Latn"],
    "tsn_Latn": ["tswana", "tsn", "tsn_Latn"],
    "tso_Latn": ["tsonga", "tso", "tso_Latn"],
    "tuk_Latn": ["turkmen", "tuk", "tuk_Latn"],
    "tum_Latn": ["tumbuka", "tum", "tum_Latn"],
    "tur_Latn": ["turkish", "tur", "tur_Latn"],
    "twi_Latn": ["twi", "twi_Latn"],
    "tzm_Tfng": ["central atlas tamazight", "tzm", "tzm_Tfng"],
    "uig_Arab": ["uyghur", "uig", "uig_Arab"],
    "ukr_Cyrl": ["ukrainian", "ukr", "ukr_Cyrl"],
    "umb_Latn": ["umbundu", "umb", "umb_Latn"],
    "urd_Arab": ["urdu", "urd", "urd_Arab"],
    "uzn_Latn": ["northern uzbek", "uzn", "uzn_Latn"],
    "vec_Latn": ["venetian", "vec", "vec_Latn"],
    "vie_Latn": ["vietnamese", "vie", "vie_Latn"],
    "war_Latn": ["waray", "war", "war_Latn"],
    "wol_Latn": ["wolof", "wol", "wol_Latn"],
    "xho_Latn": ["xhosa", "xho", "xho_Latn"],
    "ydd_Hebr": ["eastern yiddish", "ydd", "ydd_Hebr"],
    "yor_Latn": ["yoruba", "yor", "yor_Latn"],
    "yue_Hant": ["yue chinese", "yue", "yue_Hant"],
    "zho_Hans": ["chinese", "zho", "zho_Hans"],
    "zho_Hant": ["chinese", "zho", "zho_Hant"],
    "zsm_Latn": ["standard malay", "zsm", "zsm_Latn"],
    "zul_Latn": ["zulu", "zul", "zul_Latn"],
    "diq_Latn": ["southern zaza", "diq", "diq_Latn"],
    "sat_Beng": ["santali", "sat", "sat_Beng"],
}

##################################
###### LASER 3 ###################
##################################

LASER3_LANGUAGES_LIST = ["ace_Latn", "aka_Latn", "als_Latn", "amh_Ethi", "asm_Beng", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Latn", "bod_Tibt", "bug_Latn", "ceb_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dik_Latn", "diq_Latn", "dyu_Latn", "dzo_Tibt", "ewe_Latn", "fao_Latn", "fij_Latn", "fon_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", "hin_Deva", "hne_Deva", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "jav_Latn", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn", "knc_Arab", "knc_Latn", "kon_Latn", "lao_Laoo", "lij_Latn", "lim_Latn", "lin_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "prs_Arab", "quy_Latn", "run_Latn", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", "sin_Sinh", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "srd_Latn", "ssw_Latn", "sun_Latn", "swh_Latn", "szl_Latn", "tam_Taml", "taq_Latn", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "zsm_Latn", "zul_Latn"]

LASER3_LANGUAGE = build_language_names_dict(LASER3_LANGUAGES_LIST, LANGUAGE_NAMES)

##################################
###### LASER 2 ###################
##################################

LASER2_LANGUAGES_LIST = ['acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'arb_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bel_Cyrl', 'ben_Beng', 'bos_Latn', 'bul_Cyrl', 'cat_Latn', 'ces_Latn', 'ckb_Arab', 'crh_Latn', 'dan_Latn', 'deu_Latn', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'fin_Latn', 'fra_Latn', 'gle_Latn', 'glg_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jpn_Jpan', 'kab_Latn', 'kat_Geor', 'kaz_Cyrl', 'khm_Khmr', 'kmr_Latn', 'kor_Hang', 'lit_Latn', 'lvs_Latn', 'mal_Mlym', 'mar_Deva', 'mkd_Cyrl', 'plt_Latn', 'mya_Mymr', 'nld_Latn', 'nob_Latn', 'oci_Latn', 'pes_Arab', 'pol_Latn', 'por_Latn', 'ron_Latn', 'rus_Cyrl', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'snd_Arab', 'som_Latn', 'spa_Latn', 'als_Latn', 'srp_Cyrl', 'swe_Latn', 'swh_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tur_Latn', 'ukr_Cyrl', 'urd_Arab', 'uzn_Latn', 'vie_Latn', 'yue_Hant', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zsm_Latn']

LASER2_LANGUAGE = build_language_names_dict(LASER2_LANGUAGES_LIST, LANGUAGE_NAMES)

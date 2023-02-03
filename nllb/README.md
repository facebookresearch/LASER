![No Language Left Behind](nllb_laser3.png?raw=true "NLLB - LASER3")

# LASER3 - No Language Left Behind

As part of the project No Language Left Behind (NLLB) we have developed new LASER encoders, referred to here as LASER3. Each LASER3 encoder
has a particular focus language which it supports, and the full list of available LASER3 encoders can be found at [the bottom of this README](#list-of-available-laser3-encoders). 

We have also included an updated version of the original LASER encoder: LASER2. This improved model supports the same [languages](https://github.com/facebookresearch/LASER/#supported-languages) which LASER was trained on. In order to find more details on how both the LASER2 and LASER3 encoders were trained, please see [Heffernan et. al, 2022](https://arxiv.org/abs/2205.12654).

We also provide code to [train LASER3 teacher-student models](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/laser_distillation/README.md) and [stopes](https://github.com/facebookresearch/stopes), a new powerful and flexible mining library.

## Downloading encoders

To download the available encoders, please run the `download_models.sh` script within this directory. 
```
bash ./download_models.sh
```
LASER2 and all LASER3 encoders are downloaded by default. However, downloading all LASER3 encoders may take up a lot of disk space. Therefore, you may choose to select individual LASER3 encoders to download by supplying a list of available language codes (see [full list](#list-of-available-laser3-encoders)). 
For example: `bash ./download_models.sh wol_Latn zul_Latn ...`

By default, this download script will place all supported models within the calling directory.

**Note**: LASER3 encoders for each focus language are in the format: `laser3-{language_code}`.

## Embedding texts

Once encoders are downloaded, you can then begin embedding texts by following the instructions [here](/tasks/embed/README.md).

For example: `./LASER/tasks/embed/embed.sh [INFILE] [OUTFILE] wol_Latn`

## List of available LASER3 encoders

| Code | Language |
|   :---:  |  :---:  |
| ace_Latn | Acehnese (Latin script) |
| aka_Latn | Akan |
| als_Latn | Tosk Albanian |
| amh_Ethi | Amharic |
| asm_Beng | Assamese |
| awa_Deva | Awadhi |
| ayr_Latn | Central Aymara |
| azb_Arab | South Azerbaijani |
| azj_Latn | North Azerbaijani |
| bak_Cyrl | Bashkir |
| bam_Latn | Bambara |
| ban_Latn | Balinese |
| bel_Cyrl | Belarusian |
| bem_Latn | Bemba |
| ben_Beng | Bengali |
| bho_Deva | Bhojpuri |
| bjn_Latn | Banjar (Latin script) |
| bod_Tibt | Standard Tibetan |
| bug_Latn | Buginese |
| ceb_Latn | Cebuano |
| cjk_Latn | Chokwe |
| ckb_Arab | Central Kurdish |
| crh_Latn | Crimean Tatar |
| cym_Latn | Welsh |
| dik_Latn | Southwestern Dinka |
| diq_Latn | Southern Zaza |
| dyu_Latn | Dyula |
| dzo_Tibt | Dzongkha |
| ewe_Latn | Ewe |
| fao_Latn | Faroese |
| fij_Latn | Fijian |
| fon_Latn | Fon |
| fur_Latn | Friulian |
| fuv_Latn | Nigerian Fulfulde |
| gaz_Latn | West Central Oromo |
| gla_Latn | Scottish Gaelic |
| gle_Latn | Irish |
| grn_Latn | Guarani |
| guj_Gujr | Gujarati |
| hat_Latn | Haitian Creole |
| hau_Latn | Hausa |
| hin_Deva | Hindi |
| hne_Deva | Chhattisgarhi |
| hye_Armn | Armenian |
| ibo_Latn | Igbo |
| ilo_Latn | Ilocano |
| ind_Latn | Indonesian |
| jav_Latn | Javanese |
| kab_Latn | Kabyle |
| kac_Latn | Jingpho |
| kam_Latn | Kamba |
| kan_Knda | Kannada |
| kas_Arab | Kashmiri (Arabic script) |
| kas_Deva | Kashmiri (Devanagari script) |
| kat_Geor | Georgian |
| kaz_Cyrl | Kazakh |
| kbp_Latn | Kabiyè |
| kea_Latn | Kabuverdianu |
| khk_Cyrl | Halh Mongolian |
| khm_Khmr | Khmer |
| kik_Latn | Kikuyu |
| kin_Latn | Kinyarwanda |
| kir_Cyrl | Kyrgyz |
| kmb_Latn | Kimbundu |
| kmr_Latn | Northern Kurdish |
| knc_Arab | Central Kanuri (Arabic script) |
| knc_Latn | Central Kanuri (Latin script) |
| kon_Latn | Kikongo |
| lao_Laoo | Lao |
| lij_Latn | Ligurian |
| lim_Latn | Limburgish |
| lin_Latn | Lingala |
| lmo_Latn | Lombard |
| ltg_Latn | Latgalian |
| ltz_Latn | Luxembourgish |
| lua_Latn | Luba-Kasai |
| lug_Latn | Ganda |
| luo_Latn | Luo |
| lus_Latn | Mizo |
| mag_Deva | Magahi |
| mai_Deva | Maithili |
| mal_Mlym | Malayalam |
| mar_Deva | Marathi |
| min_Latn | Minangkabau (Latin script) |
| mlt_Latn | Maltese |
| mni_Beng | Meitei (Bengali script) |
| mos_Latn | Mossi |
| mri_Latn | Maori |
| mya_Mymr | Burmese |
| npi_Deva | Nepali |
| nso_Latn | Northern Sotho |
| nus_Latn | Nuer |
| nya_Latn | Nyanja |
| ory_Orya | Odia |
| pag_Latn | Pangasinan |
| pan_Guru | Eastern Panjabi |
| pap_Latn | Papiamento |
| pbt_Arab | Southern Pashto |
| pes_Arab | Western Persian |
| plt_Latn | Plateau Malagasy |
| prs_Arab | Dari |
| quy_Latn | Ayacucho Quechua |
| run_Latn | Rundi |
| sag_Latn | Sango |
| san_Deva | Sanskrit |
| sat_Beng | Santali |
| scn_Latn | Sicilian |
| shn_Mymr | Shan |
| sin_Sinh | Sinhala |
| smo_Latn | Samoan |
| sna_Latn | Shona |
| snd_Arab | Sindhi |
| som_Latn | Somali |
| sot_Latn | Southern Sotho |
| srd_Latn | Sardinian |
| ssw_Latn | Swati |
| sun_Latn | Sundanese |
| swh_Latn | Swahili |
| szl_Latn | Silesian |
| tam_Taml | Tamil |
| taq_Latn | Tamasheq (Latin script) |
| tat_Cyrl | Tatar |
| tel_Telu | Telugu |
| tgk_Cyrl | Tajik |
| tgl_Latn | Tagalog |
| tha_Thai | Thai |
| tir_Ethi | Tigrinya |
| tpi_Latn | Tok Pisin |
| tsn_Latn | Tswana |
| tso_Latn | Tsonga |
| tuk_Latn | Turkmen |
| tum_Latn | Tumbuka |
| tur_Latn | Turkish |
| twi_Latn | Twi |
| tzm_Tfng | Central Atlas Tamazight |
| uig_Arab | Uyghur |
| umb_Latn | Umbundu |
| urd_Arab | Urdu |
| uzn_Latn | Northern Uzbek |
| vec_Latn | Venetian |
| war_Latn | Waray |
| wol_Latn | Wolof |
| xho_Latn | Xhosa |
| ydd_Hebr | Eastern Yiddish |
| yor_Latn | Yoruba |
| zsm_Latn | Standard Malay |
| zul_Latn | Zulu |




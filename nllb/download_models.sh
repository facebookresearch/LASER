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
#-------------------------------------------------------
#
# This bash script installs NLLB LASER2 and LASER3 sentence encoders from Amazon s3

# default to download to current directory
mdir=$(pwd)

echo "Directory for model download: ${mdir}"

version=1  # model version

echo "Downloading networks..."

if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    if [ -f ${mdir}/${file} ] ; then
        echo " - ${mdir}/$file already downloaded";
    else
        echo " - $s3/${file}";
        wget -q $s3/${file};
    fi   
}

cd ${mdir}  # move to model directory

# available encoders
s3="https://dl.fbaipublicfiles.com/nllb/laser"

# LASER2 (download by default)
if [ ! -f ${mdir}/laser2.pt ] ; then
    echo " - $s3/laser2.pt"
    wget --trust-server-names -q https://tinyurl.com/nllblaser2
else 
    echo " - ${mdir}/laser2.pt already downloaded"
fi
download "laser2.spm"
download "laser2.cvocab"

# LASER3 models
if [ ! $# -eq 0 ]; then
    # chosen model subset from command line
    langs=$@
else
    # all available LASER3 models
    langs=(ace_Latn aka_Latn als_Latn amh_Ethi asm_Beng awa_Deva ayr_Latn azb_Arab azj_Latn bak_Cyrl bam_Latn ban_Latn bel_Cyrl \
        bem_Latn ben_Beng bho_Deva bjn_Latn bod_Tibt bug_Latn ceb_Latn cjk_Latn ckb_Arab crh_Latn cym_Latn dik_Latn diq_Latn \
        dyu_Latn dzo_Tibt ewe_Latn fao_Latn fij_Latn fon_Latn fur_Latn fuv_Latn gaz_Latn gla_Latn gle_Latn grn_Latn guj_Gujr \
        hat_Latn hau_Latn hin_Deva hne_Deva hye_Armn ibo_Latn ilo_Latn ind_Latn jav_Latn kab_Latn kac_Latn kam_Latn kan_Knda \
        kas_Arab kas_Deva kat_Geor kaz_Cyrl kbp_Latn kea_Latn khk_Cyrl khm_Khmr kik_Latn kin_Latn kir_Cyrl kmb_Latn kmr_Latn \
        knc_Arab knc_Latn kon_Latn lao_Laoo lij_Latn lim_Latn lin_Latn lmo_Latn ltg_Latn ltz_Latn lua_Latn lug_Latn luo_Latn \
        lus_Latn mag_Deva mai_Deva mal_Mlym mar_Deva min_Latn mlt_Latn mni_Beng mos_Latn mri_Latn mya_Mymr npi_Deva nso_Latn \
        nus_Latn nya_Latn ory_Orya pag_Latn pan_Guru pap_Latn pbt_Arab pes_Arab plt_Latn prs_Arab quy_Latn run_Latn sag_Latn \
        san_Deva sat_Beng scn_Latn shn_Mymr sin_Sinh smo_Latn sna_Latn snd_Arab som_Latn sot_Latn srd_Latn ssw_Latn sun_Latn \
        swh_Latn szl_Latn tam_Taml taq_Latn tat_Cyrl tel_Telu tgk_Cyrl tgl_Latn tha_Thai tir_Ethi tpi_Latn tsn_Latn tso_Latn \
        tuk_Latn tum_Latn tur_Latn twi_Latn tzm_Tfng uig_Arab umb_Latn urd_Arab uzn_Latn vec_Latn war_Latn wol_Latn xho_Latn \
        ydd_Hebr yor_Latn zsm_Latn zul_Latn)
fi

spm_langs=(amh_Ethi ayr_Latn azj_Latn bak_Cyrl bel_Cyrl bod_Tibt ckb_Arab crh_Latn dik_Latn dzo_Tibt fur_Latn \
           fuv_Latn grn_Latn kab_Latn kac_Latn kaz_Cyrl kir_Cyrl kmr_Latn lij_Latn lim_Latn lmo_Latn ltg_Latn \
           mya_Mymr pbt_Arab pes_Arab prs_Arab sat_Beng scn_Latn srd_Latn szl_Latn taq_Latn tgk_Cyrl tir_Ethi \
           tzm_Tfng vec_Latn)

for lang in ${langs[@]}; do
    download "laser3-$lang.v$version.pt";
    for spm_lang in ${spm_langs[@]}; do
        if [[ $lang == $spm_lang ]] ; then
            download "laser3-$lang.v$version.spm";
            download "laser3-$lang.v$version.cvocab";
        fi 
    done
done
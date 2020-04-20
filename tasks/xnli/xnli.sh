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
# --------------------------------------------------------
#
# bash script to downlaod and extract XNLI and multiNLI corpus

if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

xnli="XNLI-1.0"
xnli_mt="XNLI-MT-1.0"
xnli_http="https://dl.fbaipublicfiles.com/XNLI"
mnli_http="https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"

languages=("en" "fr" "es" "de" "el" "bg" "ru" "tr" "ar" "vi" "th" "zh" "hi" "sw" "ur")

edir="embed"

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

# NLI classifier params
N=200
nhid="512 384"
drop=0.3
seed=159753
bsize=128
lr=0.001

##############################################################################################
# get the XNLI dev and test corpus in 15 languages

ExtractXNLI () {
  echo "Installing XNLI"
  if [ ! -s ${xnli}/xnli.test.tsv ] ; then
      echo " - Downloading "
      wget -q  ${xnli_http}/${xnli}.zip
      echo " - unzip "
      unzip -q ${xnli}.zip
      /bin/rm -rf __MACOS ${xnli}.zip
  fi

  for lang in ${languages[@]} ; do
    for part in "dev" "test" ; do
      if [ ! -f ${edir}/xnli.${part}.prem.${lang} ] ; then
        echo " - extracting xnli.${part}.${lang}"
        tail -n +2 ${xnli}/xnli.${part}.tsv \
          | grep "^${lang}" | cut -f7 \
          > ${edir}/xnli.${part}.prem.${lang}
        tail -n +2 ${xnli}/xnli.${part}.tsv \
          | grep "^${lang}" | cut -f8 \
          > ${edir}/xnli.${part}.hyp.${lang}
        tail -n +2 ${xnli}/xnli.${part}.tsv \
          | grep "^${lang}" | cut -f2 \
          | sed -e 's/entailment/0/' -e 's/neutral/1/' -e 's/contradiction/2/' \
          > ${edir}/xnli.${part}.cl.${lang}
      fi
    done
  done
}

##############################################################################################
# https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
# MT translated data is already tokenized !

ExtractXNLI_MT () {
  echo "Installing XNLI MT"
  if [ ! -d ${xnli_mt}/multinli ] ; then
      echo " - Downloading "
      wget -q  ${xnli_http}/${xnli_mt}.zip
      echo " - unzip "
      unzip -q ${xnli_mt}.zip
      /bin/rm -rf __MACOS ${xnli_mt}.zip
  fi

  part="train"
  for lang in "en" ; do
      if [ ! -f ${edir}/multinli.${part}.prem.${lang}.gz ] ; then
        echo " - extracting ${part}.${lang}"
        tail -n +2 ${xnli_mt}/multinli/multinli.${part}.${lang}.tsv \
          | cut -f1 > ${edir}/multinli.${part}.prem.${lang}
        tail -n +2 ${xnli_mt}/multinli/multinli.${part}.${lang}.tsv \
          | cut -f2 > ${edir}/multinli.${part}.hyp.${lang}
        tail -n +2 ${xnli_mt}/multinli/multinli.${part}.${lang}.tsv \
          | cut -f3 \
          | sed -e 's/entailment/0/' -e 's/neutral/1/' -e 's/contradictory/2/' \
          > ${edir}/multinli.${part}.cl.${lang}
      fi
  done
}

##############################################################################################
# https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
# MT translated data is already tokenized !

ExtractMNLI () {
  echo "Installing MultiNLI"
  train_txt="multinli_1.0/multinli_1.0_train.txt"
  if [ ! -d ${edir} ] ; then mkdir -p ${edir}; fi

  if [ ! -f ${edir}/xnli.train.cl.en ] ; then
    echo " - Downloading"
    wget -q ${mnli_http}
    echo " - unzip"
    unzip -q multinli_1.0.zip ${train_txt}

    echo " - extracting"
    tail -n +2 ${train_txt} | cut -f6 | gzip > ${edir}/xnli.train.prem.en.gz
    tail -n +2 ${train_txt} | cut -f7 | gzip > ${edir}/xnli.train.hyp.en.gz
    tail -n +2 ${train_txt} | cut -f1 \
      | sed -e 's/entailment/0/' -e 's/neutral/1/' -e 's/contradiction/2/' \
      > ${edir}/xnli.train.cl.en
  fi
}

##############################################################################################

if [ ! -d ${edir} ] ; then mkdir -p ${edir}; fi

ExtractXNLI
ExtractMNLI

# calculate embeddings
export PYTHONPATH="$PYTHONPATH:$LASER/tools-external/jieba"
python3 xnli.py --data_dir ${edir} --lang ${languages[@]} --bpe_codes ${bpe_codes} --encoder ${encoder} --verbose

#for fr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ; do
for fr in 0.6 0.7 0.8 0.9 ; do
echo -e "\nTraining the classifier (see ${edir}/xnli.fract${fr}.log)"
python3 ${LASER}/source/nli.py -b ${edir} \
    --train xnli.train.%s.enc.en --train-labels xnli.train.cl.en \
    --dev xnli.dev.%s.enc.en --dev-labels xnli.dev.cl.en \
    --test xnli.test.%s.enc --test-labels xnli.test.cl --lang ${languages[@]} \
    --nhid ${nhid[@]} --dropout ${drop} --bsize ${bsize} \
    --seed ${seed} --lr ${lr} --nepoch ${N} \
    --cross-lingual \
    --fraction $fr \
    --save-outputs ${edir}/xnli.fract${fr}.outputs \
    --gpu 1 > ${edir}/xnli.fract${fr}.log
done

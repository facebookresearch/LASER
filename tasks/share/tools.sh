#!/bin/bash
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# This module contains several bash functions which are useful to calculate
# sentence embeddings

if [ -z ${LASER+x} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

bdir="${LASER}"
mlencdir="${bdir}/mlenc"
tools="${bdir}/tools"
tools_ext="${bdir}/tools-external"
mdir="${bdir}/models"
shdir="${bdir}/tasks/share"

# Moses tokenizer and normalization tools
tok="tok.lc"
moses_tok="${tools_ext}/moses-tokenizer/tokenizer.perl"
moses_nthreads=20 # threaded tokenization
moses_detok="${tools_ext}/moses-tokenizer/detokenizer.perl"
moses_norm_punct="${tools_ext}/moses-tokenizer/normalize-punctuation.perl"
moses_deesc="${tools_ext}/moses-tokenizer/deescape-special-chars.perl"
moses_lc="${tools_ext}/moses-tokenizer/lowercase.perl"
# JIEBA tokenizer for Chinese
jieba_tokenizer="${tools_ext}/jieba3k/jieba_tokenizer.py"

# The sentence embeddings have been trained on corpora which
# contain almost no contractions (eg. "don't") but only the
# full form (eg "do not")
# we normlizae the input texts so that they are similar to the
# training data.
ndir="${tools}/normalize"


###################################################################
#
# Some utility functions
#
###################################################################

MKDIR () {
  df=$1
  if [ ! -d ${df} ] ; then
    echo " - creating directory ${df}"
    mkdir -p ${df}
  fi
}


###################################################################
#
# Apply various normalizations and tokenize texts
#
###################################################################

Tokenize () {
  fname=${1}
  lang=$2
  if [ -f ${fname}.${lang} ] ; then
    CAT=cat; term=""; ZIP="cat"
  elif [ -f ${fname}.${lang}.gz ] ; then
    CAT=zcat; term=".gz"; ZIP="gzip"
  else
    echo "ERROR in tokenization: input file ${fname}.${lang} doesn't exist"
    exit
  fi

  if [ ! -f ${fname}.${tok}.${lang}${term} ] ; then
    if [ ${lang} == "zh" ] ; then
      echo " - tokenize `basename ${fname}.${lang}` with jieba"
      export PYTHONPATH="jieba"
      tmpf=/tmp/jieba.${USER}.$$
      ${CAT} ${fname}.${lang}${term} \
        | ${moses_deesc} \
        | ${moses_norm_punct} -l ${lang} \
        > ${tmpf}.in
      python -u ${jieba_tokenizer} --inp_file ${tmpf}.in --out_file ${tmpf}.out
      cat ${tmpf}.out \
        | ${moses_lc} \
        | sed -f ${ndir}/contractions_${lang}.sed \
        | ${ZIP} > ${fname}.${ftok}.${lang}${term}
      /bin/rm ${tmpf}.*
    else
      echo " - tokenize `basename ${fname}.${lang}` with Moses"
      ${CAT} ${fname}.${lang}${term} \
          | ${moses_deesc} \
          | ${moses_norm_punct} -l ${lang} \
          | ${moses_tok} -q -no-escape -threads ${moses_nthreads} -l ${lang} \
          | ${moses_lc} \
          | sed -f ${ndir}/contractions_${lang}.sed \
          | ${ZIP} > ${fname}.${tok}.${lang}${term}
    fi
  fi
}


###################################################################
#
# Simple approach to split very long sentences into multiple lines.
# The texts must be already tokenized.
#
###################################################################

SplitLines () {
  fname=$1 # base filename
  lang=$2  # language
  if [ -f ${fname}.${lang} ] ; then
    CAT=cat; term=""; ZIP="cat"
  elif [ -f ${fname}.${lang}.gz ] ; then
    CAT=zcat; term=".gz"; ZIP="gzip"
  else
    echo "ERROR in SplitLines: input file ${fname}.${lang} doesn't exist"
    exit
  fi

  if="${fname}.${lang}${term}"
  if [ ! -f ${fname}.split.${lang}${term} ] ; then
    echo -n " - split `basename ${if}`, lines/max_words:"
    echo -n " `${CAT} ${if} | wc -l `/`${CAT} ${if} | awk '{print NF}' | sort -n | tail -1` -> "
    ${CAT} ${if} \
      | awk '{ printf("%d\t",NR); \
               for (i=1;i<=NF;i++) { \
                 printf(" %s",$i); \
                 if ($i=="." && i!=NF) printf("\n%d\t",NR) \
               } \
              printf("\n") }' \
       | ${ZIP} > ${fname}.tmp.${lang}${term}

    ${CAT} ${fname}.tmp.${lang}${term} | cut -d'	' -f1 > ${fname}.sid.${lang}${term}
    ${CAT} ${fname}.tmp.${lang}${term} | cut -d'	' -f2 > ${fname}.split.${lang}${term}
    # TODO: handle empty lines
    if2="${fname}.split.${lang}${term}"
    echo " `${CAT} ${if2} | wc -l `/`${CAT} ${if2} | awk '{print NF}' | sort -n | tail -1`"
    /bin/rm ${fname}.tmp.${lang}${term}
  fi
}


###################################################################
#
# Join the sentence embeddings corresponding to lines which have been split
# before
#
###################################################################

JoinLines()
{
  pname=$1 # base filename

  if=`printf ${pname} "split.enc"`
  sid=`printf ${pname} "sid"`
  of=`printf ${pname} "join.enc"`
  if [ ! -f $of ] ; then
    echo " - join embedded sentences into `basename $of`"
    python ${shdir}/combine_embed.py --inp $if --sid $sid --out $of > ${of}.log
  fi
}

###################################################################
#
# Calculate sentence embedding
#
###################################################################

Embed()
{
  fname=$1
  lang=$2
  mod_name=$3
  maxlen=$4
  gpu=$5

  if [ ${mod_name} == "eparl20kj" ] ; then
    msg="embed `basename ${fname}` in language ${lang} with eparl bpej20k"
    bpe_codes="${mdir}/binarize/ep7.9langs-v1.bpej20k.codes.9xx"
    hash_table="${mdir}/binarize/ep7.9langs-v1.bpej20k.bin.9xx"
    model="${mdir}/networks/blstm.ep7.9langs-v1.bpej20k.model.py"
  else
    echo "ERROR: model ${mod_name} is unknown"
    exit
  fi

  if [ -f ${fname}.${lang} ] ; then
    CAT=cat; term=""; ZIP="cat"
  elif [ -f ${fname}.${lang}.gz ] ; then
    CAT=zcat; term=".gz"; ZIP="gzip"
  else
    echo "ERROR: input file doesn't exist"
    exit
  fi

  if [ ! -f ${fname}.enc.${lang} ] ; then
    echo " - ${msg}"
    ${CAT} ${fname}.${lang}${term} \
      | python ${mlencdir}/mlenc.py --gpu ${gpu} \
               --verbose  1 \
               --max_len ${maxlen} \
               --bpe_code ${bpe_codes} \
               --hash_table ${hash_table} \
               --model ${model}  \
               --output_bpe ${fname}.bpe.${lang} \
               --output_enc ${fname}.enc.${lang} \
      > ${fname}.enc.${lang}.log

    if [ `grep "skipped:" ${fname}.enc.${lang}.log | sed -e 's/=/ /g' | awk '{N=NF-6; print $N}'` -gt 0 ]; then
      echo "ERROR: skipped sentences during embedding"
      exit
    fi
  fi
}

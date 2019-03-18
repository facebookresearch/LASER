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
# bash script to mine for bitexts in the BUCC corpus


if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

# general config
bucc="bucc2018"
data="."
xdir=${data}/downloaded	# tar files as distrubuted by the BUCC evaluation
ddir=${data}/${bucc}	# raw texts of BUCC
edir=${data}/embed	# normalized texts and embeddings
langs=("fr" "de" "ru" "zh")
ltrg="en"		# English is always the 2nd language

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"


###################################################################
#
# Extract files with labels and texts from the BUCC corpus
#
###################################################################

GetData () {
  fn1=$1; fn2=$2; lang=$3
  outf="${edir}/${bucc}.${lang}-${ltrg}.${fn2}"
  for ll  in ${ltrg} ${lang} ; do
    inf="${ddir}/${fn1}.${ll}"
    if [ ! -f ${outf}.txt.${ll} ] ; then
      echo " - extract files ${outf} in ${ll}"
      cat ${inf} | cut -f1 > ${outf}.id.${ll}
      cat ${inf} | cut -f2 > ${outf}.txt.${ll}
    fi
  done
}

ExtractBUCC () {
  slang=$1
  tlang=${ltrg}

  pushd ${data} > /dev/null
  if [ ! -d ${ddir}/${slang}-${tlang} ] ; then
    for tf in ${xdir}/${bucc}-${slang}-${tlang}.*.tar.bz2 ; do
      echo " - extract from tar `basename ${tf}`"
      tar jxf $tf
    done
  fi

  GetData "${slang}-${tlang}/${slang}-${tlang}.sample" "dev" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.training" "train" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.test" "test" ${slang}
  popd > /dev/null
}


###################################################################
#
# Tokenize and Embed
#
###################################################################

Embed () {
  ll=$2
  txt="$1.txt.${ll}"
  enc="$1.enc.${ll}"
  if [ ! -s ${enc} ] ; then
    cat ${txt} | python3 ${LASER}/source/embed.py \
      --encoder ${encoder} \
      --token-lang ${ll} \
      --bpe-codes ${bpe_codes} \
      --output ${enc} \
      --verbose 
  fi
}


###################################################################
#
# Mine for bitexts
#
###################################################################

Mine () {
  bn=$1
  l1=$2
  l2=$3
  cand="${bn}.candidates.tsv"
  if [ ! -s ${cand} ] ; then
    python3 ${LASER}/source/mine_bitexts.py \
       ${bn}.txt.${l1} ${bn}.txt.${l2} \
       --src-lang ${l1} --trg-lang ${l2} \
       --src-embeddings ${bn}.enc.${l1} --trg-embeddings ${bn}.enc.${l2} \
       --unify --mode mine --retrieval max --margin ratio -k 4  \
       --output ${cand} \
       --verbose --gpu
  fi
}


###################################################################
#
# Main loop
#
###################################################################

echo -e "\nProcessing BUCC data in ${data}"

# create output directories
for d in ${ddir} ${edir} ; do
  mkdir -p ${d}
done

for lsrc in ${langs[@]} ; do
  ExtractBUCC ${lsrc}

  # Tokenize and embed train 
  bname="${bucc}.${lsrc}-${ltrg}"
  part="${bname}.train"
  Embed ${edir}/${part} ${lsrc} ${encoder} ${bpe_codes}
  Embed ${edir}/${part} ${ltrg} ${encoder} ${bpe_codes}

  # mine for texts in train
  Mine ${edir}/${part} ${lsrc} ${ltrg}

  # optimize threshold on BUCC training data and provided gold alignments
  if [ ! -s ${part}.log ] ; then
    python3 bucc.py \
      --src-lang ${lsrc} --trg-lang ${ltrg} \
      --bucc-texts ${edir}/${part}.txt \
      --bucc-ids ${edir}/${part}.id \
      --candidates ${edir}/${part}.candidates.tsv \
      --gold ${ddir}/${lsrc}-${ltrg}/${lsrc}-${ltrg}.training.gold \
      --verbose \
      | tee ${part}.log
  fi

  # Tokenize and embed test 
  part="${bname}.test"
  Embed ${edir}/${part} ${lsrc} ${encoder} ${bpe_codes}
  Embed ${edir}/${part} ${ltrg} ${encoder} ${bpe_codes}

  # mine for texts in test
  Mine ${edir}/${part} ${lsrc} ${ltrg}

  # extract test bitexts for treshhold optimized on train
  th=`grep 'best threshold' ${bname}.train.log | sed -e 's/[=:]/ /g' | awk '{print $4}'`
  extracted="${edir}/${part}.extracted.tsv"
  if [ ! -s ${extracted} ] ; then
    python3 bucc.py \
      --src-lang ${lsrc} --trg-lang ${ltrg} \
      --bucc-texts ${edir}/${part}.txt \
      --bucc-ids ${edir}/${part}.id \
      --candidates ${edir}/${part}.candidates.tsv \
      --threshold ${th} --output ${extracted} \
      --verbose
  fi
done

# Bonus: extract bitexts with English alignments
# using a (conservative) threshold of 1.1
# All the data is supposed to be already tokenized

th=1.1
for lsrc in ${langs[@]} ; do
  for ltrg in ${langs[@]} ; do
    if [ ${lsrc} != 'en' -a ${ltrg} != "en" -a ${lsrc} != ${ltrg} ] ; then
      bitext="${bucc}.${lsrc}-${ltrg}.train.extracted.th${th}.csv"
      if [ ! -s ${bitext} ] ; then
        echo "Extracting bitexts for ${lsrc}-${ltrg}"
        python3 ${LASER}/source/mine_bitexts.py \
          ${edir}/${bucc}.${lsrc}-en.train.txt.${lsrc} \
          ${edir}/${bucc}.${ltrg}-en.train.txt.${ltrg} \
          --src-lang ${lsrc} --trg-lang ${ltrg} \
          --src-embeddings ${edir}/${bucc}.${lsrc}-en.train.enc.${lsrc} \
          --trg-embeddings ${edir}/${bucc}.${ltrg}-en.train.enc.${ltrg} \
          --unify --mode mine --retrieval max --margin ratio -k 4  \
          --output ${bitext} --threshold ${th} \
          --verbose --gpu
      fi
    fi
  done
done

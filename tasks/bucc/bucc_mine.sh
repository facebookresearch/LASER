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
# bash script to mine for bitexts in the BUCC data


# general config
bucc="bucc2018"
data="."
xdir=${data}/downloaded	# tar files as distrubuted by the BUCC evaluation
ddir=${data}/${bucc}	# raw texts of BUCC
edir=${data}/embed	# normalized texts and embeddings
langs=("fr" "de" )
lang0="en"		# English is always the 2nd language

# parameters
id="id"
txt="txt"
maxlen=2500		# maxmimum number of words after sentence splitting and BPE
mod="eparl20kj"		# which BLSTM model to use
gpu=0			# which GPU to use, use -1 for CPU mode
k=1			# we only request the nearest neighbors

# include common functions
source ../share/tools.sh

# where to find LASER tools
mldir="${LASER}/tools"


###################################################################
#
# Calculate cosine difference between all sentences in the
# source and target language.
# This takes about 30-40 min on CPU (multithreaded)
#
# We use L2 distance on normalized embeddings which is identical
# to cosine distance
###################################################################

CalcDiffs () {
  fname=$1
  slang=$2
  tlang=$3

  if [ ! -f ${fname}.cos.k${k}.dist ] ; then
    echo "Calculate all distances for pair ${slang}-${tlang}"
    time python ${mldir}/faiss/calc_dist_pairwise.py --fname ${fname} \
      --lang1 ${slang} --lang2 ${tlang} \
      --norm --k ${k}
  fi
}


###################################################################
#
# Find the optimal threshold on the multilingual distance
# We usually do this on the training data and then apply
# the best threshold on the test data.
#
# Simple implementation using standard Linux command line tools...
#
###################################################################

TuneThreshold () {
  fname=$1
  slang=$2
  tlang=$3
  align_ref=$4

  align="${fname}.align"	# alignments for current threshold
  thresh="${fname}.thresh"	# result file with scores for all threshold

  echo "Optimizing threshold for pair ${slang}-${tlang}"
  if [ ! -f ${thresh} ] ; then
    ntgt=`wc -l < ${align_ref}`
    echo " - calculating scores for all thresholds against ${ntgt} refs"
    cat < /dev/null > ${thresh}
    th=200
    while [ $th -le 1000 ] ; do
      echo -ne " - threshold ${th}\t" >> ${thresh}
      # find all pairs which have a distance smaller than the threshold
      # add 1 to index  since BUCC: 1..N, but Python: 0...N-1
      paste ${fname}.{dist,idx} \
        | awk -vTH=${th} -vL1=${slang} -vL2=${tlang} \
              '{ if ($1<TH/1000) printf("%s-%09d\t%s-%09d\n", L1, NR, L2, $2+1)}' \
        > ${align}

      # Compare with gold alignments
      # and calculate precision, recall and F-score
      nsrc=`wc -l < ${align}`
      nok=`comm -12 ${align} ${align_ref} | wc -l`
      echo "${nok} ${nsrc} ${ntgt}" \
        | awk '{P=$1/$2; R=$1/$3; printf("ok=%d P=%.2f R=%.2f F1=%.2f\n", $1, 100*P, 100*R, 200*P*R/(P+R)) }' \
        >> ${thresh}
      # increase threshold
      let th=th+1
    done
  fi

  echo -en " - precision:\t"
  sed -e 's/=/ /g' -e 's/hold /hold 0./' ${thresh} | sort -nr -k7 | head -1
  echo -en " - recall:\t"
  sed -e 's/=/ /g' -e 's/hold /hold 0./' ${thresh} | sort -nr -k9 | head -1
  echo -en " - F1:\t\t"
  sed -e 's/=/ /g' -e 's/hold /hold 0./' ${thresh} | sort -nr -k11 | head -1
}


###################################################################
#
# Extract bitexts for a given threshold
#
# Output format of file bucc2018....extract (tab seperated)
# dist	Sentence_in_language_1    Sentence_in_language_2

###################################################################

ExtractBitext () {
  fname=$1
  l1=$2
  l2=$3
  th=$4
  text1=$5
  text2=$6

  align="${fname}.align"	# alignments for current threshold
  thresh="${fname}.thresh"	# result file with scores for all threshold
  echo "Getting alignments"
  echo " - file `basename $fname`"
  echo " - threshold $th"
  if [ ! -f ${align} ] ; then
    paste ${fname}.{dist,idx} \
      | awk -vTH=${th} -vL1=${l1} -vL2=${l2} \
            '{ if ($1<TH) printf("%s-%09d\t%s-%09d\t%f\n", L1, NR, L2, $2+1, $1)}' \
      > ${align}
  fi
  echo " - found: `wc -l < ${align}` lines"

  if [ -f ${align}.extract ] ; then
    echo " - already extacted into file ${align}.extract"
  else 
    echo " - extracting texts into file ${align}.extract"

    # simple implementation which can be easily optimzied ...
    cat < /dev/null > ${align}.extract
    while read l ; do
      echo $l | awk '{printf("%f\t", $3)}' >> ${align}.extract
      key=`echo $l | awk '{print $1}'`
      echo -ne "`grep $key ${text1} | cut -f2`\t" >> ${align}.extract
      key=`echo $l | awk '{print $2}'`
      echo -e "`grep $key ${text2} | cut -f2`\t" >> ${align}.extract
    done < ${align}
  fi
}


###################################################################
#
# Main loop
#
###################################################################

# create output directories
for d in ${ddir} ${edir} ; do
  MKDIR ${d}
done

echo -e "\nProcessing BUCC data in ${data}"
for l in ${langs[@]} ; do
  bname="${bucc}.${l}-${lang0}"

  # Calculate all pairwsie differences and find best threshold
  # for the provided gold alignments
  # The order of the languages is important !
  # for BUCC, we need to create the index on English
  CalcDiffs ${edir}/${bname}.train.${txt}.${tok}.enc  ${l} ${lang0}
  TuneThreshold ${edir}/${bname}.train.${txt}.${tok}.enc.cos.k${k} \
                ${l} ${lang0} \
                ${ddir}/${l}-${lang0}/${l}-${lang0}.training.gold
done


###################################################################
#
# Extract the bitexts for the test data using the optimized threshold
#
###################################################################

echo "Working on the test data:"
for l in ${langs[@]} ; do
  part="${bucc}.${l}-${lang0}.test"
  Tokenize ${edir}/${part}.${txt} ${l}
  Embed ${edir}/${part}.${txt}.${tok} ${l} ${mod} ${maxlen} ${gpu}
  CalcDiffs ${edir}/${part}.${txt}.${tok}.enc ${l} ${lang0}
done

l1="fr"; l2="en"; thopt=0.519
ExtractBitext "${edir}/${bucc}.${l1}-${l2}.test.${txt}.${tok}.enc.cos.k${k}" ${l1} ${l2} ${thopt} \
              "${ddir}/${l1}-${l2}/${l1}-${l2}.test.${l1}" \
              "${ddir}/${l1}-${l2}/${l1}-${l2}.test.${l2}" 

l1="de"; l2="en"; thopt=0.499
ExtractBitext "${edir}/${bucc}.${l1}-${l2}.test.${txt}.${tok}.enc.cos.k${k}" ${l1} ${l2} ${thopt} \
              "${ddir}/${l1}-${l2}/${l1}-${l2}.test.${l1}" \
              "${ddir}/${l1}-${l2}/${l1}-${l2}.test.${l2}" 


###################################################################
#
# Bonus: extract French-German parallel data
#
# There are no gold alignments for that language pair
# and we use a threshold of 0.500, similar to the values
# which were optimized for Fr-En and De-En
#
###################################################################

l1="fr"; l2="de"; l0="en"
part="${bucc}.${l1}-${l2}.test"
echo -e "\nExtracting bitext for ${part}"

pushd ${edir} > /dev/null
ln -s "${bucc}.${l1}-${l0}.test.${txt}.${tok}.enc.${l1}" "${part}.${txt}.${tok}.enc.${l1}"
ln -s "${bucc}.${l2}-${l0}.test.${txt}.${tok}.enc.${l2}" "${part}.${txt}.${tok}.enc.${l2}"
popd > /dev/null

CalcDiffs "${edir}/${part}.${txt}.${tok}.enc" ${l1} ${l2}

thresh=0.500
ExtractBitext "${edir}/${part}.${txt}.${tok}.enc.cos.k${k}" ${l1} ${l2} ${thresh} \
              "${ddir}/${l1}-${l0}/${l1}-${l0}.test.${l1}" \
              "${ddir}/${l2}-${l0}/${l2}-${l0}.test.${l2}" 

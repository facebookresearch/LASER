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
# bash script to train and evaluate classifiers on the MLDoc corpus


# include common functions
source ../share/tools.sh

# where to find LASER tools
mldir="${LASER}/tools"

# data specification
data="."
edir=${data}/embed	# normalized texts and embeddings
tok="tok.lc"
lbl="lbl"
txt="txt"
bname="mldoc"				# id used is all created files
langs=("de" "en" "es" "fr" "it")	# languages for which training data is available

# parameters
nepoch=100	# number of training epochs
gpu=0		# which GPU to use, use -1 for CPU mode

# output files
logdir="log"

###################################################################
#
# Train the classifier with the specified parameters
#
###################################################################

Train () {
  ltrn=$1
  ldev=$2
  bs=$3
  nh=$4
  dp=$5
  sd=$6
  ne=$7

  fn="${bname}.${ltrn}-${ldev}.nh${nh}-do${dp}-bsize${bs}-is${sd}-ep${ne}"

  if [ ! -f ${logdir}/${fn}.log ] ; then
    python ${mldir}/classify_sent.py \
           --gpu ${gpu} \
           --base_dir ${edir} \
           --train "${bname}.train1000.${txt}.${tok}.join.enc.${ltrn}" --train_labels "${bname}.train1000.${lbl}.${ldev}" \
           --dev   "${bname}.dev.${txt}.${tok}.join.enc.${ldev}" --dev_labels "${bname}.dev.${lbl}.${ldev}" \
           --test  "${bname}.test.${txt}.${tok}.join.enc" --test_labels "${bname}.test.${lbl}"  \
           --langs "de" --langs "en" --langs "es" --langs "fr" --langs "it" \
           --nepoch ${ne} --seed ${sd} --bsize ${bs} --nhid ${nh} --dropout ${dp} \
           > ${logdir}/${fn}.log
           #--save ${fn}.net
    echo " - ${fn} `grep 'Best Dev' ${logdir}/${fn}.log`"
  fi
}


###################################################################
#
# Parse log files to find best parameters
#
###################################################################

FindBestDev() {
  bdir=$1
  ltrn=$2
  ldev=$3

  fn=`grep "Best Dev" ${logdir}/${bname}.${ltrn}-${ldev}.*log | sort -nr -k5 | head -1 | sed -e 's/log:.*/log/'`
  dev=`tail -6 ${fn} | head -1 | awk '{print $5}'`
  printf "%s" ${dev}
  for l in ${langs[@]} ; do
    grep "Eval Test lang ${l}" ${fn} | awk '{printf("\t%s",$10)}'
  done
  echo -e "\t$fn"
}



###################################################################
#
# Find best classifiers for ZERO-SHORT transfer
# sweep over parameters optimizing
# on the dev corpus of the TRAINING language
#
###################################################################

OptimizeZeroShot () {
  nepeoch=100

  for lang_train in ${langs[@]} ; do
    lang_dev=${lang_train}
    echo "Sweep over parameters to optimize model for ${lang_train}-${lang_dev}"
    for bsize in 8 12 16 32 ; do
      for nhid in 4 5 6 7 8 10 12 14 16 16 18 20 ; do
        for drop in 0.1 0.2 0.3 0.4 ; do
          for seed in 123456789 14081967 3615 ; do

            Train ${lang_train} ${lang_dev} ${bsize} ${nhid} ${drop} ${seed} ${nepoch}

          done
        done
      done

      # get best performance on DEV of training language
      # and print accuracies on all test languges
      FindBestDev "." ${lang_train} ${lang_dev}
    done
  done
}


###################################################################
#
# Train classifier for zero-short transfer
# using optimized parameters
#
###################################################################

MKDIR ${logdir}

# parameters:  lang_train lang_dev bsize nhid dropout seed nbepochs
echo "Training models with optimized settings"
Train "de" "de"  8  5 0.3 123456789 100
Train "en" "en" 32  4 0.4      3615 100
Train "es" "es"  8 12 0.1  14081967 100
Train "fr" "fr"  8 12 0.2  14081967 100
Train "it" "it" 16  7 0.4      3615 100


###################################################################
#
# Run parameter sweep to find optimal settings for all languages
# This can take a couple of hours to train many models for each language
#
###################################################################

#OptimizeZeroShot

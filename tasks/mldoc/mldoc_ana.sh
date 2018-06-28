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
# bash script to parse log-files


logdir="log"
langs=("de" "en" "es" "fr" "it")

echo -e "Dev\tde\ten\tes\tfr\tit\tParams"
for ltrn in ${langs[@]} ; do
  for ldev in $ltrn; do

    fn=`grep -H "Best Dev" ${logdir}/mldoc.${ltrn}-${ldev}.*log | sort -nr -k5 | head -1 | sed -e 's/log:.*/log/'`
    dev=`tail -6 ${fn} | head -1 | awk '{print $5}'`
    printf "%s" ${dev}
    for l in ${langs[@]} ; do
      grep "Eval Test lang ${l}" ${fn} | awk '{printf("\t%s",$10)}'
    done
    echo -e "\t$fn"
  done
done

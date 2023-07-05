#!/usr/bin/python
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
# Helper functions for tokenization and BPE

import os
import sys
import logging
import sentencepiece as spm
from sacremoses import MosesPunctNormalizer, MosesDetokenizer

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("preprocess")

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'

class LaserTokenizer:          
    def tokenize(inp_fname, out_fname, spm_model_file, lang='en',
                lower_case=True, descape=False,
                verbose=False, over_write=False, gzip=False):
        assert lower_case, 'lower case is needed by all the models'
        if not os.path.isfile(out_fname):
            cat = 'zcat ' if gzip else 'cat '
            if verbose:
                logger.info('SPM processing {} {} {}'
                    .format(os.path.basename(inp_fname),
                            '(gzip)' if gzip else '',
                            '(de-escaped)' if descape else ''))

            assert os.path.isfile(spm_model_file), f'SPM model {spm_model_file} not found'
            
            with open(inp_fname, 'r', encoding='utf-8') as file:
                    text = file.read()

            mpn = MosesPunctNormalizer(lang=lang)
            md = MosesDetokenizer()

            # Preprocessing steps
            sentence_text = "".join(c for c in text if c.isprintable())
            sentence_text = mpn.normalize(sentence_text)
            if descape:
                sentence_text = md.unescape_xml(text=sentence_text)
            sentence_text = sentence_text.lower()

            # SentencePiece encoding
            spm_model = spm.SentencePieceProcessor(model_file=spm_model_file)
            spm_processed = " ".join(spm_model.encode(sentence_text, out_type=str))

            with open(out_fname, 'w', encoding='utf-8') as file:
                file.write(''.join(spm_processed))

        elif not over_write and verbose:
            logger.info('SPM encoded file {} exists already'
                .format(os.path.basename(out_fname)))

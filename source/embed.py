#!/usr/bin/python3
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
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code


import re
import os
import tempfile
import sys
import time
import argparse
import numpy as np
import logging
from collections import namedtuple
from subprocess import run
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


from lib.text_processing import Token, BPEfastApply, SPMApply

from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder,
)
from fairseq.data.dictionary import Dictionary
from fairseq.modules import LayerNorm

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger('embed')

def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


class SentenceEncoder:
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        spm_vocab=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
    ):
        if verbose:
            logger.info(f"loading encoder: {model_path}")
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        if "params" in state_dict:
            self.encoder = LaserLstmEncoder(**state_dict["params"])
            self.encoder.load_state_dict(state_dict["model"])
            self.dictionary = state_dict["dictionary"]
            self.prepend_bos = False
            self.left_padding = False
        else:
            self.encoder = LaserTransformerEncoder(state_dict, spm_vocab)
            self.dictionary = self.encoder.dictionary.indices
            self.prepend_bos = state_dict["cfg"]["model"].prepend_bos
            self.left_padding = state_dict["cfg"]["model"].left_pad_source
        del state_dict
        self.bos_index = self.dictionary["<s>"] = 0
        self.pad_index = self.dictionary["<pad>"] = 1
        self.eos_index = self.dictionary["</s>"] = 2
        self.unk_index = self.dictionary["<unk>"] = 3

        if fp16:
            self.encoder.half()
        if self.use_cuda:
            if verbose:
                logger.info("transfer encoder to GPU")
            self.encoder.cuda()
        self.encoder.eval()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            sentemb = self.encoder(tokens, lengths)["sentemb"]
        embeddings = sentemb.detach().cpu().numpy()
        return embeddings

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        if self.prepend_bos:
            ids = torch.LongTensor(ntokens + 2)
            ids[0] = self.bos_index
            for i, token in enumerate(tokens):
                ids[i + 1] = self.dictionary.get(token, self.unk_index)
            ids[ntokens + 1] = self.eos_index
        else:
            ids = torch.LongTensor(ntokens + 1)
            for i, token in enumerate(tokens):
                ids[i] = self.dictionary.get(token, self.unk_index)
            ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            if not self.left_padding:
                for i in range(len(tokens)):
                    toks[i, : tokens[i].shape[0]] = tokens[i]
            else:
                for i in range(len(tokens)):
                    toks[i, -tokens[i].shape[0] :] = tokens[i]
            return (
                Batch(srcs=None, tokens=toks, lengths=torch.LongTensor(lengths)),
                indices,
            )

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and (
                (self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens)
                or (self.max_sentences is not None and nsentences == self.max_sentences)
            ):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class HuggingFaceEncoder():
    def __init__(self, encoder_name: str, verbose=False):
        from sentence_transformers import SentenceTransformer
        encoder = f"sentence-transformers/{encoder_name}"
        if verbose:
            logger.info(f"loading HuggingFace encoder: {encoder}")
        self.encoder = SentenceTransformer(encoder)

    def encode_sentences(self, sentences):
        return self.encoder.encode(sentences)


class LaserTransformerEncoder(TransformerEncoder):
    def __init__(self, state_dict, vocab_path):
        self.dictionary = Dictionary.load(vocab_path)
        if any(
            k in state_dict["model"]
            for k in ["encoder.layer_norm.weight", "layer_norm.weight"]
        ):
            self.dictionary.add_symbol("<mask>")
        cfg = state_dict["cfg"]["model"]
        self.sentemb_criterion = cfg.sentemb_criterion
        self.pad_idx = self.dictionary.pad_index
        self.bos_idx = self.dictionary.bos_index
        embed_tokens = Embedding(
            len(self.dictionary), cfg.encoder_embed_dim, self.pad_idx,
        )
        super().__init__(cfg, self.dictionary, embed_tokens)
        if "decoder.version" in state_dict["model"]:
            self._remove_decoder_layers(state_dict)
        if "layer_norm.weight" in state_dict["model"]:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.load_state_dict(state_dict["model"])

    def _remove_decoder_layers(self, state_dict):
        for key in list(state_dict["model"].keys()):
            if not key.startswith(
                (
                    "encoder.layer_norm",
                    "encoder.layers",
                    "encoder.embed",
                    "encoder.version",
                )
            ):
                del state_dict["model"][key]
            else:
                renamed_key = key.replace("encoder.", "")
                state_dict["model"][renamed_key] = state_dict["model"].pop(key)

    def forward(self, src_tokens, src_lengths):
        encoder_out = super().forward(src_tokens, src_lengths)
        if isinstance(encoder_out, dict):
            x = encoder_out["encoder_out"][0]  # T x B x C
        else:
            x = encoder_out[0]
        if self.sentemb_criterion == "cls":
            cls_indices = src_tokens.eq(self.bos_idx).t()
            sentemb = x[cls_indices, :]
        else:
            padding_mask = src_tokens.eq(self.pad_idx).t().unsqueeze(-1)
            if padding_mask.any():
                x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)
            sentemb = x.max(dim=0)[0]
        return {"sentemb": sentemb}


class LaserLstmEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embed_dim=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        left_pad=True,
        padding_value=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            "sentemb": sentemb,
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": encoder_padding_mask
            if encoder_padding_mask.any()
            else None,
        }


def load_model(
    encoder: str,
    spm_model: str,
    bpe_codes: str,
    hugging_face=False,
    verbose=False,
    **encoder_kwargs
) -> Union[SentenceEncoder, HuggingFaceEncoder]:
    if hugging_face:
        return HuggingFaceEncoder(encoder, verbose=verbose)
    if spm_model:
        spm_vocab = str(Path(spm_model).with_suffix(".cvocab"))
        if verbose:
            logger.info(f"spm_model: {spm_model}")
            logger.info(f"spm_cvocab: {spm_vocab}")
    else:
        spm_vocab = None
    return SentenceEncoder(
        encoder, spm_vocab=spm_vocab, verbose=verbose, **encoder_kwargs
    )


def EncodeLoad(args):
    args.buffer_size = max(args.buffer_size, 1)
    assert (
        not args.max_sentences or args.max_sentences <= args.buffer_size
    ), "--max-sentences/--batch-size cannot be larger than --buffer-size"

    print(" - loading encoder", args.encoder)
    return SentenceEncoder(
        args.encoder,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
        cpu=args.cpu,
        verbose=args.verbose,
    )


def EncodeTime(t):
    t = int(time.time() - t)
    if t < 1000:
        return "{:d}s".format(t)
    else:
        return "{:d}m{:d}s".format(t // 60, t % 60)


# Encode sentences (existing file pointers)
def EncodeFilep(
    encoder, inp_file, out_file, buffer_size=10000, fp16=False, verbose=False
):
    n = 0
    t = time.time()
    for sentences in buffered_read(inp_file, buffer_size):
        encoded = encoder.encode_sentences(sentences)
        if fp16:
            encoded = encoded.astype(np.float16)
        encoded.tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            logger.info("encoded {:d} sentences".format(n))
    if verbose:
        logger.info(f"encoded {n} sentences in {EncodeTime(t)}")



# Encode sentences (file names)
def EncodeFile(
    encoder,
    inp_fname,
    out_fname,
    buffer_size=10000,
    fp16=False,
    verbose=False,
    over_write=False,
    inp_encoding="utf-8",
):
    # TODO :handle over write
    if not os.path.isfile(out_fname):
        if verbose:
            logger.info(
                "encoding {} to {}".format(
                    inp_fname if len(inp_fname) > 0 else "stdin", out_fname,
                )
            )
        fin = (
            open(inp_fname, "r", encoding=inp_encoding, errors="surrogateescape")
            if len(inp_fname) > 0
            else sys.stdin
        )
        fout = open(out_fname, mode="wb")
        EncodeFilep(
            encoder, fin, fout, buffer_size=buffer_size, fp16=fp16, verbose=verbose
        )
        fin.close()
        fout.close()
    elif not over_write and verbose:
        logger.info("encoder: {} exists already".format(os.path.basename(out_fname)))


# Load existing embeddings
def EmbedLoad(fname, dim=1024, verbose=False):
    x = np.fromfile(fname, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(" - Embeddings: {:s}, {:d}x{:d}".format(fname, x.shape[0], dim))
    return x


# Get memory mapped embeddings
def EmbedMmap(fname, dim=1024, dtype=np.float32, verbose=False):
    nbex = int(os.path.getsize(fname) / dim / np.dtype(dtype).itemsize)
    E = np.memmap(fname, mode="r", dtype=dtype, shape=(nbex, dim))
    if verbose:
        print(" - embeddings on disk: {:s} {:d} x {:d}".format(fname, nbex, dim))
    return E


def embed_sentences(
    ifname: Path,
    output: Path,
    encoder: Union[SentenceEncoder, HuggingFaceEncoder] = None,
    encoder_path: Path = None,
    hugging_face = False,
    token_lang: Optional[str] = "--",
    bpe_codes: Optional[Path] = None,
    spm_lang: Optional[str] = "en",
    spm_model: Optional[Path] = None,
    verbose: bool = False,
    buffer_size: int = 10000,
    max_tokens: int = 12000,
    max_sentences: Optional[int] = None,
    cpu: bool = False,
    fp16: bool = False,
    sort_kind: str = "quicksort",
):
    assert encoder or encoder_path, "Provide initialised encoder or encoder_path"
    buffer_size = max(buffer_size, 1)
    assert (
        not max_sentences or max_sentences <= buffer_size
    ), "--max-sentences/--batch-size cannot be larger than --buffer-size"

    assert not (bpe_codes and spm_model), "Cannot specify both spm and bpe"

    if encoder_path:
        encoder = load_model(
            encoder_path,
            spm_model,
            bpe_codes,
            verbose=verbose,
            hugging_face=hugging_face,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind=sort_kind,
            cpu=cpu,
        )
    if not ifname:
        ifname = ""  # default to stdin
    with tempfile.TemporaryDirectory() as tmpdir:
        if token_lang != "--":
            tok_fname = os.path.join(tmpdir, "tok")
            Token(
                ifname,
                tok_fname,
                lang=token_lang,
                romanize=True if token_lang == "el" else False,
                lower_case=True,
                gzip=False,
                verbose=verbose,
                over_write=False,
            )
            ifname = tok_fname

        if bpe_codes:
            if ifname == "":  # stdin
                ifname = os.path.join(tmpdir, "no_tok")
                run(f'cat > {ifname}', shell=True)
            bpe_fname = os.path.join(tmpdir, "bpe")
            BPEfastApply(
                ifname, bpe_fname, bpe_codes, verbose=verbose, over_write=False
            )
            ifname = bpe_fname

        if spm_model:
            spm_fname = os.path.join(tmpdir, "spm")
            SPMApply(
                ifname,
                spm_fname,
                spm_model,
                lang=spm_lang,
                lower_case=True,
                verbose=verbose,
                over_write=False,
            )
            ifname = spm_fname

        EncodeFile(
            encoder,
            ifname,
            output,
            verbose=verbose,
            over_write=False,
            buffer_size=buffer_size,
            fp16=fp16,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Embed sentences")
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input text file",
    )
    parser.add_argument("--encoder", type=str, required=True, help="encoder to be used")
    parser.add_argument(
        "--token-lang",
        type=str,
        default="--",
        help="Perform tokenization with given language ('--' for no tokenization)",
    )
    parser.add_argument(
        "--bpe-codes", type=str, default=None, help="Apply BPE using specified codes"
    )
    parser.add_argument(
        "--spm-lang", type=str, default="en", help="Apply SPM using specified language"
    )
    parser.add_argument(
        "--spm-model", type=str, default=None, help="Apply SPM using specified model"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")

    parser.add_argument(
        "-o", "--output", required=True, help="Output sentence embeddings"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Buffer size (sentences)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=12000,
        help="Maximum number of tokens to process in a batch",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences to process in a batch",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store embedding matrices in fp16 instead of fp32",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--sort-kind",
        type=str,
        default="quicksort",
        choices=["quicksort", "mergesort"],
        help="Algorithm used to sort batch by length",
    )
    parser.add_argument(
        "--use-hugging-face", action="store_true", help="Use a HuggingFace sentence transformer"
    )

    args = parser.parse_args()
    embed_sentences(
        ifname=args.input,
        encoder_path=args.encoder,
        token_lang=args.token_lang,
        bpe_codes=args.bpe_codes,
        spm_lang=args.spm_lang,
        hugging_face=args.use_hugging_face,
        spm_model=args.spm_model,
        verbose=args.verbose,
        output=args.output,
        buffer_size=args.buffer_size,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        cpu=args.cpu,
        fp16=args.fp16,
        sort_kind=args.sort_kind,
    )

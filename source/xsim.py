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
# Tool to calculate multilingual similarity error rate (xSIM)

import faiss
import numpy as np
import typing as tp
import os
from enum import Enum


class Margin(Enum):
    RATIO = "ratio"
    DISTANCE = "distance"
    ABSOLUTE = "absolute"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


def xSIM(
    x: tp.Union[str, np.ndarray],
    y: tp.Union[str, np.ndarray],
    margin: str = Margin.RATIO.value,
    k: int = 4,
    dim: int = 1024,
    fp16: bool = False,
    eval_text: str = None,
) -> tp.Tuple[int, int]:
    assert Margin.has_value(margin), f"Margin type: {margin}, is not supported."
    if not isinstance(x, np.ndarray):
        x = _load_embeddings(x, dim, fp16)
    if not isinstance(y, np.ndarray):
        y = _load_embeddings(y, dim, fp16)
    # calculate xSIM error
    return calculate_error(x, y, margin, k, eval_text)


def _load_embeddings(infile: str, dim: int, fp16: bool = False) -> np.ndarray:
    assert os.path.isfile(infile), f"file: {infile} does not exist."
    emb = np.fromfile(infile, dtype=np.float16 if fp16 else np.float32)
    num_examples = emb.shape[0] // dim
    emb.resize(num_examples, dim)
    if fp16:
        emb = emb.astype(np.float32)  # faiss currently only supports fp32
    return emb


def _score_margin(
    Dxy: np.ndarray,
    Ixy: np.ndarray,
    Ax: np.ndarray,
    Ay: np.ndarray,
    margin: str,
    k: int,
) -> np.ndarray:
    nbex = Dxy.shape[0]
    scores = np.zeros((nbex, k))
    for i in range(nbex):
        for j in range(k):
            jj = Ixy[i, j]
            a = Dxy[i, j]
            b = (Ax[i] + Ay[jj]) / 2
            if margin == Margin.RATIO.value:
                scores[i, j] = a / b
            else:  # distance margin
                scores[i, j] = a - b
    return scores


def _score_knn(x: np.ndarray, y: np.ndarray, k: int, margin: str) -> np.ndarray:
    nbex, dim = x.shape
    # create index
    idx_x = faiss.IndexFlatIP(dim)
    idx_y = faiss.IndexFlatIP(dim)
    # L2 normalization needed for cosine distance
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)
    idx_x.add(x)
    idx_y.add(y)
    if margin == Margin.ABSOLUTE.value:
        scores, indices = idx_y.search(x, 1)
    else:
        # return cosine similarity and indices of k closest neighbors
        Cos_xy, Idx_xy = idx_y.search(x, k)
        Cos_yx, Idx_yx = idx_x.search(y, k)

        # average cosines
        Avg_xy = Cos_xy.mean(axis=1)
        Avg_yx = Cos_yx.mean(axis=1)

        scores = _score_margin(Cos_xy, Idx_xy, Avg_xy, Avg_yx, margin, k)

        # find best
        best = scores.argmax(axis=1)
        indices = np.zeros((nbex, 1), dtype=np.int32)
        for i in range(nbex):
            indices[i] = Idx_xy[i, best[i]]
    return indices


def calculate_error(
    x: np.ndarray,
    y: np.ndarray,
    margin: str = None,
    k: int = 4,
    eval_text: str = None,
) -> tp.Tuple[int, int]:
    assert (
        x.shape == y.shape
    ), f"number of source {x.shape} / target {y.shape} shapes mismatch"
    nbex = x.shape[0]

    # for each x calculate the highest scoring neighbor from y
    closest_neighbor = _score_knn(x, y, k, margin)

    if eval_text:  # calc textual error
        lines = open(eval_text, encoding="utf-8", errors="surrogateescape").readlines()
        err = 0
        for ex in range(nbex):
            if lines[ex] != lines[closest_neighbor[ex, 0]]:
                err += 1
    else:  # calc index error
        ref = np.linspace(0, nbex - 1, nbex).astype(int)  # [0, nbex)
        err = nbex - np.equal(closest_neighbor.reshape(nbex), ref).astype(int).sum()
    return err, nbex

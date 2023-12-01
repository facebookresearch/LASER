# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for various tasks such as document classification,
# and bitext filtering
#
# --------------------------------------------------------
#
# Tool to calculate the dual approach multilingual similarity error rate (P-xSIM)

import typing as tp
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from stopes.eval.auto_pcp.audio_comparator import Comparator, get_model_pred
from xsim import Margin, score_margin


def get_neighbors(
    x: np.ndarray, y: np.ndarray, k: int, margin: str
) -> tp.Tuple[np.ndarray, np.ndarray, int]:
    x_copy = x.astype(np.float32).copy()
    y_copy = y.astype(np.float32).copy()
    nbex, dim = x.shape
    # create index
    idx_x = faiss.IndexFlatIP(dim)
    idx_y = faiss.IndexFlatIP(dim)
    # L2 normalization needed for cosine distance
    faiss.normalize_L2(x_copy)
    faiss.normalize_L2(y_copy)
    idx_x.add(x_copy)
    idx_y.add(y_copy)
    if margin == Margin.ABSOLUTE.value:
        scores, indices = idx_y.search(x_copy, k)
    else:
        # return cosine similarity and indices of k closest neighbors
        Cos_xy, Idx_xy = idx_y.search(x_copy, k)
        Cos_yx, Idx_yx = idx_x.search(y_copy, k)

        # average cosines
        Avg_xy = Cos_xy.mean(axis=1)
        Avg_yx = Cos_yx.mean(axis=1)

        scores = score_margin(Cos_xy, Idx_xy, Avg_xy, Avg_yx, margin, k)
        indices = Idx_xy
    return scores, indices, nbex


def get_cosine_scores(src_emb: np.ndarray, neighbor_embs: np.ndarray) -> np.ndarray:
    assert src_emb.shape[0] == neighbor_embs.shape[1]
    src_embs = np.repeat(
        np.expand_dims(src_emb, axis=0), neighbor_embs.shape[0], axis=0
    )
    cosine_scores = cosine_similarity(src_embs, neighbor_embs).diagonal()
    return cosine_scores


def get_comparator_scores(
    src_emb: np.ndarray,
    neighbor_embs: np.ndarray,
    comparator_model: tp.Any,
    symmetrize_comparator: bool,
) -> np.ndarray:
    src_embs = np.repeat(
        np.expand_dims(src_emb, axis=0), neighbor_embs.shape[0], axis=0
    )
    a = torch.from_numpy(src_embs).unsqueeze(1)  # restore depth dim
    b = torch.from_numpy(neighbor_embs).unsqueeze(1)
    res = get_comparator_preds(a, b, comparator_model, symmetrize_comparator)
    scores_softmax = softmax(res)
    return np.array(scores_softmax)


def get_comparator_preds(
    src_emb: np.ndarray, tgt_emb: np.ndarray, model: tp.Any, symmetrize: bool
):
    preds = (
        get_model_pred(
            model,
            src=src_emb[:, 0],
            mt=tgt_emb[:, 0],
            use_gpu=model.use_gpu,
            batch_size=1,
        )[:, 0]
        .cpu()
        .numpy()
    )
    if symmetrize:
        preds2 = (
            get_model_pred(
                model,
                src=tgt_emb[:, 0],
                mt=src_emb[:, 0],
                use_gpu=model.use_gpu,
                batch_size=1,
            )[:, 0]
            .cpu()
            .numpy()
        )
        preds = (preds2 + preds) / 2
    return preds


def get_blended_predictions(
    alpha: float,
    nbex: int,
    margin_scores: np.ndarray,
    x_aux: np.ndarray,
    y_aux: np.ndarray,
    neighbor_indices: np.ndarray,
    comparator_model: tp.Optional[tp.Any] = None,
    symmetrize_comparator: bool = False,
) -> list[int]:
    predictions = []
    for src_index in range(nbex):
        neighbors = neighbor_indices[src_index]
        neighbor_embs = y_aux[neighbors].astype(np.float32)
        src_emb = x_aux[src_index].astype(np.float32)
        aux_scores = (
            get_comparator_scores(
                src_emb, neighbor_embs, comparator_model, symmetrize_comparator
            )
            if comparator_model
            else get_cosine_scores(src_emb, neighbor_embs)
        )
        assert margin_scores[src_index].shape == aux_scores.shape
        blended_scores = alpha * margin_scores[src_index] + (1 - alpha) * aux_scores
        blended_neighbor_idx = blended_scores.argmax()
        predictions.append(neighbors[blended_neighbor_idx])
    return predictions


def PxSIM(
    x: np.ndarray,
    y: np.ndarray,
    x_aux: np.ndarray,
    y_aux: np.ndarray,
    alpha: float,
    margin: str = Margin.RATIO.value,
    k: int = 16,
    comparator_path: tp.Optional[Path] = None,
    symmetrize_comparator: bool = False,
) -> tp.Tuple[int, int, list[int]]:
    """
    Parameters
    ----------
    x : np.ndarray
        source-side embedding array
    y : np.ndarray
        target-side embedding array
    x_aux : np.ndarray
        source-side embedding array using auxiliary model
    y_aux : np.ndarray
        target-side embedding array using auxiliary model
    alpha : int
        parameter to weight blended score
    margin : str
        margin scoring function (e.g. ratio, absolute, distance)
    k : int
        number of neighbors in k-nn search
    comparator_path : Path
        path to AutoPCP model config
    symmetrize_comparator : bool
        whether to symmetrize the comparator predictions

    Returns
    -------
    err : int
        Number of errors
    nbex : int
        Number of examples
    preds : list[int]
        List of (index-based) predictions
    """
    assert Margin.has_value(margin), f"Margin type: {margin}, is not supported."
    comparator_model = Comparator.load(comparator_path) if comparator_path else None
    # get margin-based nearest neighbors
    margin_scores, neighbor_indices, nbex = get_neighbors(x, y, k=k, margin=margin)
    preds = get_blended_predictions(
        alpha,
        nbex,
        margin_scores,
        x_aux,
        y_aux,
        neighbor_indices,
        comparator_model,
        symmetrize_comparator,
    )
    err = sum([idx != pred for idx, pred in enumerate(preds)])
    print(f"P-xSIM error: {100 * (err / nbex):.2f}")
    return err, nbex, preds


def load_embeddings(
    infile: Path, dim: int, fp16: bool = False, numpy_header: bool = False
) -> np.ndarray:
    assert infile.exists(), f"file: {infile} does not exist."
    if numpy_header:
        return np.load(infile)
    emb = np.fromfile(infile, dtype=np.float16 if fp16 else np.float32)
    num_examples = emb.shape[0] // dim
    emb.resize(num_examples, dim)
    if fp16:
        emb = emb.astype(np.float32)  # faiss currently only supports fp32
    return emb


def run(
    src_emb: Path,
    tgt_emb: Path,
    src_aux_emb: Path,
    tgt_aux_emb: Path,
    alpha: float,
    margin: str = Margin.RATIO.value,
    k: int = 16,
    emb_fp16: bool = False,
    aux_emb_fp16: bool = False,
    emb_dim: int = 1024,
    aux_emb_dim: int = 1024,
    numpy_header: bool = False,
    comparator_path: tp.Optional[Path] = None,
    symmetrize_comparator: bool = False,
    prediction_savepath: tp.Optional[Path] = None,
) -> None:
    x = load_embeddings(src_emb, emb_dim, emb_fp16, numpy_header)
    y = load_embeddings(tgt_emb, emb_dim, emb_fp16, numpy_header)
    x_aux = load_embeddings(src_aux_emb, aux_emb_dim, aux_emb_fp16, numpy_header)
    y_aux = load_embeddings(tgt_aux_emb, aux_emb_dim, aux_emb_fp16, numpy_header)
    assert (x.shape == y.shape) and (x_aux.shape == y_aux.shape)
    _, _, preds = PxSIM(
        x, y, x_aux, y_aux, alpha, margin, k, comparator_path, symmetrize_comparator
    )
    if prediction_savepath:
        with open(prediction_savepath, "w") as outf:
            for pred in preds:
                print(pred, file=outf)


if __name__ == "__main__":
    import func_argparse

    func_argparse.main()

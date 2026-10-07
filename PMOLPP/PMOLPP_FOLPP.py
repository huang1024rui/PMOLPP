#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The function of the file: 构建邻接图并使用薄QR分解实现快速正交LPP（FOLPP）。
"""

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist


def _check_data(Data):
    Data = np.asarray(Data, dtype=float)
    if Data.ndim != 2:
        raise ValueError("Data必须是二维数组。")
    if Data.shape[0] < 2 or Data.shape[1] < 1:
        raise ValueError("Data至少需要2个样本和1个特征。")
    if not np.all(np.isfinite(Data)):
        raise ValueError("Data中不能包含NaN或无穷值。")
    return Data


def build_knn_graph(Data, n_neighbors, t=1e7):
    Data = _check_data(Data)
    n_samples = Data.shape[0]
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1 or n_neighbors >= n_samples:
        raise ValueError("n_neighbors必须位于[1, n_samples-1]。")
    if t <= 0:
        raise ValueError("热核参数t必须大于0。")

    # SECTION: K近邻热核图
    distance_square = cdist(Data, Data, metric="sqeuclidean")
    np.fill_diagonal(distance_square, np.inf)
    neighbor_indices = np.argpartition(
        distance_square, kth=n_neighbors - 1, axis=1
    )[:, :n_neighbors]

    W_directed = np.zeros((n_samples, n_samples), dtype=float)
    row_indices = np.repeat(np.arange(n_samples), n_neighbors)
    col_indices = neighbor_indices.reshape(-1)
    W_directed[row_indices, col_indices] = np.exp(
        -distance_square[row_indices, col_indices] / float(t)
    )

    # NOTE: 采用并集方式对称化，确保无向图且主对角线为0。
    W = np.maximum(W_directed, W_directed.T)
    np.fill_diagonal(W, 0.0)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return W, D, L


def _normalize_qr_sign(projection, qr_r):
    projection = projection.copy()
    qr_r = qr_r.copy()
    for i in range(projection.shape[1]):
        max_index = int(np.argmax(np.abs(projection[:, i])))
        column_sign = np.sign(projection[max_index, i])
        if column_sign == 0:
            column_sign = 1.0
        projection[:, i] = projection[:, i] * column_sign
        qr_r[i, :] = qr_r[i, :] * column_sign
    return projection, qr_r


def folpp(Data, n_neighbors, n_dims, t=1e7, regularization=1e-8, epsilon=1e-12):
    Data = _check_data(Data)
    n_features = Data.shape[1]
    n_dims = int(n_dims)
    if n_dims < 1 or n_dims > n_features:
        raise ValueError("n_dims必须位于[1, n_features]。")
    if regularization <= 0 or epsilon <= 0:
        raise ValueError("regularization和epsilon必须大于0。")

    W, D, L = build_knn_graph(Data, n_neighbors=n_neighbors, t=t)
    XLX = Data.T @ L @ Data
    XDX = Data.T @ D @ Data

    # SECTION: 对XDX白化后求解对称广义特征值问题
    scale = max(float(np.trace(XDX)) / n_features, 1.0)
    XDX = (XDX + XDX.T) / 2.0
    XDX_regular = XDX + regularization * scale * np.eye(n_features)
    XLX = (XLX + XLX.T) / 2.0

    # NOTE: 只在未正则化XDX的有效数据子空间中求解，避免高维情形选中
    # 正则化人为产生的零空间方向。
    eig_B, vec_B = scipy.linalg.eigh(XDX)
    rank_tolerance = max(
        epsilon,
        float(np.max(np.abs(eig_B)))
        * max(XDX.shape)
        * np.finfo(float).eps,
    )
    positive = eig_B > rank_tolerance
    effective_rank = int(np.sum(positive))
    actual_dims = min(n_dims, effective_rank)

    if effective_rank == 0:
        candidate = np.eye(n_features, min(n_dims, n_features))
        projection, qr_r = np.linalg.qr(candidate, mode="reduced")
        projection, qr_r = _normalize_qr_sign(projection, qr_r)
        selected_eigenvalues = np.zeros(projection.shape[1], dtype=float)
        generalized_residual = 0.0
        return {
            "projection": projection,
            "candidate_projection": candidate,
            "qr_r": qr_r,
            "W": W,
            "D": D,
            "L": L,
            "eigenvalues": selected_eigenvalues,
            "effective_rank": 0,
            "generalized_residual": generalized_residual,
        }

    B_inverse_half_reduced = vec_B[:, positive] / np.sqrt(
        eig_B[positive] + regularization * scale
    )
    whitened = B_inverse_half_reduced.T @ XLX @ B_inverse_half_reduced
    whitened = (whitened + whitened.T) / 2.0
    eig_values, eig_vectors = scipy.linalg.eigh(whitened)
    selected_indices = np.argsort(eig_values)[:actual_dims]
    selected_eigenvalues = eig_values[selected_indices]
    candidate = B_inverse_half_reduced @ eig_vectors[:, selected_indices]

    # NOTE: FOLPP关键步骤，使用薄QR一次得到正交投影矩阵。
    projection, qr_r = np.linalg.qr(candidate, mode="reduced")
    projection, qr_r = _normalize_qr_sign(projection, qr_r)

    residual = XLX @ candidate - (
        XDX_regular @ candidate
    ) * selected_eigenvalues.reshape(1, -1)
    residual_denominator = max(
        float(np.linalg.norm(XLX @ candidate, ord="fro")), epsilon
    )
    generalized_residual = float(
        np.linalg.norm(residual, ord="fro") / residual_denominator
    )

    return {
        "projection": projection,
        "candidate_projection": candidate,
        "qr_r": qr_r,
        "W": W,
        "D": D,
        "L": L,
        "eigenvalues": selected_eigenvalues,
        "effective_rank": effective_rank,
        "generalized_residual": generalized_residual,
    }

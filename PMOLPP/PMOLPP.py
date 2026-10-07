#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The function of the file: 实现PMNMSL、MSDF和F2M2组成的PMOLPP模型。
"""

import numpy as np

from .PMOLPP_FOLPP import folpp
from .PMOLPP_GeoMLE import estimate_intrinsic_dimension


def select_significant_features(projection, eta_sig):
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 2 or projection.shape[0] == 0 or projection.shape[1] == 0:
        raise ValueError("projection必须是非空二维数组。")
    if not 0 < eta_sig <= 1:
        raise ValueError("eta_sig必须位于(0, 1]。")
    threshold = eta_sig * np.max(np.abs(projection), axis=0, keepdims=True)
    return np.where(np.any(np.abs(projection) >= threshold, axis=1))[0]


def calculate_laplacian_energies(embedding, laplacian):
    embedding = np.asarray(embedding, dtype=float)
    laplacian = np.asarray(laplacian, dtype=float)
    if embedding.ndim != 2 or laplacian.ndim != 2:
        raise ValueError("embedding和laplacian必须是二维数组。")
    if laplacian.shape != (embedding.shape[0], embedding.shape[0]):
        raise ValueError("laplacian维数必须与样本数一致。")
    centered = embedding - np.mean(embedding, axis=0, keepdims=True)
    energies = np.sum(centered * (laplacian @ centered), axis=0)
    # NOTE: 拉普拉斯矩阵半正定，微小负值仅来自浮点舍入。
    return np.maximum(energies, 0.0)


def normalize_fusion_weights(consistency_scores, epsilon=1e-12):
    consistency_scores = np.asarray(consistency_scores, dtype=float)
    if consistency_scores.ndim != 1 or consistency_scores.size == 0:
        raise ValueError("consistency_scores必须是非空一维数组。")
    if epsilon <= 0:
        raise ValueError("epsilon必须大于0。")
    if not np.all(np.isfinite(consistency_scores)) or np.any(consistency_scores < 0):
        raise ValueError("consistency_scores必须是非负有限值。")
    inverse_scores = 1.0 / np.maximum(consistency_scores, epsilon)
    return inverse_scores / inverse_scores.sum()


class PMOLPP:
    def __init__(
        self,
        n_neighbors=None,
        t=1e7,
        eta_sig=0.8,
        eta_stop=0.5,
        max_layers=None,
        regularization=1e-8,
        random_state=None,
    ):
        self.n_neighbors = n_neighbors
        self.t = t
        self.eta_sig = eta_sig
        self.eta_stop = eta_stop
        self.max_layers = max_layers
        self.regularization = regularization
        self.random_state = random_state
        self.is_fitted = False

    @staticmethod
    def _check_data(Data, require_two_samples=False):
        Data = np.asarray(Data, dtype=float)
        if Data.ndim != 2:
            raise ValueError("Data必须是二维数组。")
        minimum_samples = 2 if require_two_samples else 1
        if Data.shape[0] < minimum_samples or Data.shape[1] < 1:
            raise ValueError(
                "Data至少需要{}个样本和1个特征。".format(minimum_samples)
            )
        if not np.all(np.isfinite(Data)):
            raise ValueError("Data中不能包含NaN或无穷值。")
        return Data

    def _check_parameters(self, n_samples):
        if not 0 < self.eta_sig <= 1:
            raise ValueError("eta_sig必须位于(0, 1]。")
        if self.eta_stop <= 0:
            raise ValueError("eta_stop必须大于0。")
        if self.t <= 0 or self.regularization <= 0:
            raise ValueError("t和regularization必须大于0。")
        if self.max_layers is not None and int(self.max_layers) < 1:
            raise ValueError("max_layers必须为正整数或None。")

        if self.n_neighbors is None:
            n_neighbors = int(np.floor(np.sqrt(n_samples)))
        else:
            n_neighbors = int(self.n_neighbors)
        return int(np.clip(n_neighbors, 1, n_samples - 1))

    def fit(self, Data_train):
        Data_train = self._check_data(Data_train, require_two_samples=True)
        n_samples, n_features = Data_train.shape
        self.n_neighbors_ = self._check_parameters(n_samples)
        self.n_features_in_ = n_features
        self.layers_ = []
        self.fusion_weights_ = np.array([], dtype=float)
        self.consistency_scores_ = np.array([], dtype=float)
        self.is_fitted = False

        current_feature_indices = np.arange(n_features, dtype=int)
        layer_index = 0

        # SECTION: PMNMSL逐层学习
        while current_feature_indices.size > 0:
            if self.max_layers is not None and layer_index >= int(self.max_layers):
                break

            # NOTE: 每层均从原始训练数据直接重构，避免逐层投影误差传播。
            Data_current = Data_train[:, current_feature_indices]
            n_dims, dimension_info = estimate_intrinsic_dimension(
                Data_current,
                random_state=self.random_state,
            )
            n_dims = int(np.clip(n_dims, 1, Data_current.shape[1]))
            folpp_result = folpp(
                Data_current,
                n_neighbors=self.n_neighbors_,
                n_dims=n_dims,
                t=self.t,
                regularization=self.regularization,
            )

            projection = folpp_result["projection"]
            n_dims = projection.shape[1]
            embedding = Data_current @ projection
            significant_local = select_significant_features(
                projection,
                eta_sig=self.eta_sig,
            )
            significant_original = current_feature_indices[significant_local]
            remaining_original = np.setdiff1d(
                current_feature_indices,
                significant_original,
                assume_unique=True,
            )

            energies = calculate_laplacian_energies(
                embedding,
                folpp_result["L"],
            )
            total_energy = float(np.sum(energies))
            if total_energy <= np.finfo(float).eps:
                msdf_score = 0.0
            else:
                msdf_score = float(energies[0] / total_energy)
            accepted = bool(msdf_score < self.eta_stop)

            if layer_index == 0:
                self.original_laplacian_ = folpp_result["L"].copy()

            # NOTE: 首层不合格时保留退化输出，但不继续递归。
            retain_layer = accepted or layer_index == 0
            if retain_layer:
                diagnostics = {
                    "eigenvalues": folpp_result["eigenvalues"].copy(),
                    "candidate_projection": folpp_result[
                        "candidate_projection"
                    ].copy(),
                    "qr_r": folpp_result["qr_r"].copy(),
                    "generalized_residual": float(
                        folpp_result["generalized_residual"]
                    ),
                    "orthogonality_error": float(
                        np.linalg.norm(
                            projection.T @ projection - np.eye(projection.shape[1]),
                            ord="fro",
                        )
                    ),
                }
                self.layers_.append(
                    {
                        "layer_index": layer_index,
                        "feature_indices": current_feature_indices.copy(),
                        "significant_local_indices": significant_local.copy(),
                        "significant_original_indices": significant_original.copy(),
                        "remaining_original_indices": remaining_original.copy(),
                        "projection": projection.copy(),
                        "intrinsic_dimension": int(n_dims),
                        "msdf_score": float(msdf_score),
                        "accepted_by_msdf": bool(accepted),
                        "dimension_info": dimension_info,
                        "folpp_diagnostics": diagnostics,
                    }
                )

            if not accepted:
                break
            if remaining_original.size == 0:
                break
            if np.array_equal(remaining_original, current_feature_indices):
                break

            current_feature_indices = remaining_original
            layer_index += 1

        if len(self.layers_) == 0:
            raise RuntimeError("PMOLPP未能生成有效层。")

        # SECTION: F2M2一致性加权融合
        consistency_scores = []
        for layer in self.layers_:
            layer_embedding = (
                Data_train[:, layer["feature_indices"]] @ layer["projection"]
            )
            layer_energies = calculate_laplacian_energies(
                layer_embedding,
                self.original_laplacian_,
            )
            consistency_score = float(np.mean(layer_energies))
            consistency_scores.append(consistency_score)

        self.consistency_scores_ = np.asarray(consistency_scores, dtype=float)
        self.fusion_weights_ = normalize_fusion_weights(self.consistency_scores_)
        for layer, consistency_score, fusion_weight in zip(
            self.layers_, self.consistency_scores_, self.fusion_weights_
        ):
            layer["consistency_score"] = float(consistency_score)
            layer["fusion_weight"] = float(fusion_weight)

        self.n_output_features_ = int(
            sum(layer["projection"].shape[1] for layer in self.layers_)
        )
        self.is_fitted = True
        return self

    def transform(self, Data):
        if not self.is_fitted:
            raise RuntimeError("请先使用训练数据调用fit。")
        Data = self._check_data(Data)
        if Data.shape[1] != self.n_features_in_:
            raise ValueError("Data的特征数与训练数据不一致。")

        embedding_list = []
        for layer, fusion_weight in zip(self.layers_, self.fusion_weights_):
            layer_embedding = (
                Data[:, layer["feature_indices"]] @ layer["projection"]
            )
            embedding_list.append(float(fusion_weight) * layer_embedding)
        return np.concatenate(embedding_list, axis=1)

    def fit_transform(self, Data_train):
        return self.fit(Data_train).transform(Data_train)

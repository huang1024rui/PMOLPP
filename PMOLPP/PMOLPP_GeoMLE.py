#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The function of the file: 封装GeoMLE并在数值异常时使用矩阵秩回退。
"""

import numpy as np
import pandas as pd

from GLPP_LE.geomle import geomle


def _matrix_rank_fallback(Data, fallback_reason):
    centered_data = Data - np.mean(Data, axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered_data))
    rank = int(np.clip(rank, 1, Data.shape[1]))
    info = {
        "method": "matrix_rank_fallback",
        "estimated_dimension": rank,
        "fallback_reason": str(fallback_reason),
    }
    return rank, info


def estimate_intrinsic_dimension(Data, random_state=None):
    Data = np.asarray(Data, dtype=float)
    if Data.ndim != 2:
        raise ValueError("Data必须是二维数组。")
    if Data.shape[0] < 2 or Data.shape[1] < 1:
        raise ValueError("Data至少需要2个样本和1个特征。")
    if not np.all(np.isfinite(Data)):
        raise ValueError("Data中不能包含NaN或无穷值。")

    n_samples, n_features = Data.shape
    if n_features == 1:
        return 1, {
            "method": "single_feature",
            "estimated_dimension": 1,
        }

    k2 = min(40, n_samples - 2)
    k1 = min(10, max(3, k2 - 2))
    try:
        if k2 <= k1:
            raise ValueError("有效样本数不足以执行GeoMLE。")

        # NOTE: 保留原GeoMLE实现，仅在外层完成输入适配和结果保护。
        Data_frame = pd.DataFrame(Data)
        dimension_values = np.asarray(
            geomle(
                Data_frame,
                k1=k1,
                k2=k2,
                random_state=random_state,
            ),
            dtype=float,
        )
        dimension_mean = float(np.mean(dimension_values))
        if not np.isfinite(dimension_mean):
            raise ValueError("GeoMLE返回了非有限结果。")

        dimension = int(np.rint(dimension_mean))
        dimension = int(np.clip(dimension, 1, n_features))
        info = {
            "method": "GeoMLE",
            "estimated_dimension": dimension,
            "raw_mean": dimension_mean,
            "k1": int(k1),
            "k2": int(k2),
        }
        return dimension, info
    except Exception as error:
        return _matrix_rank_fallback(Data, error)

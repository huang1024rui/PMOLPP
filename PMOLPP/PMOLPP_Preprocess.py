#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The function of the file: 使用训练集统计量完成PMOLPP数据预处理。

Data_type = 0: 保持原数据
Data_type = 1: 减去训练集特征均值
Data_type = 2: 除以训练集特征标准差
Data_type = 3: 除以训练集全局极差
Data_type = 4: 除以训练集Frobenius范数
"""

import numpy as np


class PMOLPP_Preprocess:
    def __init__(self, Data_type=0, epsilon=1e-12):
        self.Data_type = Data_type
        self.epsilon = epsilon
        self.is_fitted = False

    @staticmethod
    def _check_data(Data):
        Data = np.asarray(Data, dtype=float)
        if Data.ndim != 2:
            raise ValueError("Data必须是二维数组。")
        if Data.shape[0] == 0 or Data.shape[1] == 0:
            raise ValueError("Data不能为空。")
        if not np.all(np.isfinite(Data)):
            raise ValueError("Data中不能包含NaN或无穷值。")
        return Data

    def fit(self, Data_train):
        # NOTE: 所有归一化统计量只允许从训练集计算。
        Data_train = self._check_data(Data_train)
        if self.Data_type not in {0, 1, 2, 3, 4}:
            raise ValueError("Data_type必须取0、1、2、3或4。")
        if self.epsilon <= 0:
            raise ValueError("epsilon必须大于0。")

        self.n_features_in_ = Data_train.shape[1]
        if self.Data_type == 1:
            self.feature_mean_ = np.mean(Data_train, axis=0)
        elif self.Data_type == 2:
            feature_std = np.std(Data_train, axis=0)
            self.feature_std_ = np.where(feature_std > self.epsilon, feature_std, 1.0)
        elif self.Data_type == 3:
            Data_range = float(np.max(Data_train) - np.min(Data_train))
            self.Data_range_ = Data_range if Data_range > self.epsilon else 1.0
        elif self.Data_type == 4:
            Data_norm = float(np.linalg.norm(Data_train, ord="fro"))
            self.Data_norm_ = Data_norm if Data_norm > self.epsilon else 1.0

        self.is_fitted = True
        return self

    def transform(self, Data):
        if not self.is_fitted:
            raise RuntimeError("请先使用训练数据调用fit。")

        Data = self._check_data(Data)
        if Data.shape[1] != self.n_features_in_:
            raise ValueError("Data的特征数与训练数据不一致。")

        # SECTION: 使用fit阶段保存的统计量变换新数据
        if self.Data_type == 0:
            Data_new = Data.copy()
        elif self.Data_type == 1:
            Data_new = Data - self.feature_mean_
        elif self.Data_type == 2:
            Data_new = np.true_divide(Data, self.feature_std_)
        elif self.Data_type == 3:
            Data_new = np.true_divide(Data, self.Data_range_)
        else:
            Data_new = np.true_divide(Data, self.Data_norm_)
        return Data_new

    def fit_transform(self, Data_train):
        return self.fit(Data_train).transform(Data_train)

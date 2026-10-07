#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PMOLPP 学习参数配置。

这些参数通过仅使用训练数据的内部验证学习得到，避免将测试数据用于参数选择。
"""

import numbers
from collections.abc import Mapping


_PMOLPP_PARAMETER_CONFIG = {
    # SECTION: DNA, two_stage, selection_seeds=(10, 54, 24)
    1: {
        300: {"eta_sig": 0.7, "eta_stop": 0.1, "max_layers": None, "t": 10000},
        350: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 100000},
        400: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 1000000},
        450: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 10000},
    },
    # SECTION: Spambase, two_stage, selection_seeds=(10, 54, 24)
    6: {
        200: {"eta_sig": 0.6, "eta_stop": 0.3, "max_layers": None, "t": 1000000},
        400: {"eta_sig": 0.5, "eta_stop": 0.3, "max_layers": None, "t": 10000},
        600: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 1000000},
        800: {"eta_sig": 0.95, "eta_stop": 0.1, "max_layers": None, "t": 1000000},
    },
    # SECTION: Msplice, two_stage, selection_seeds=(10, 54, 24)
    7: {
        300: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 10000},
        400: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 10000},
        500: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 10000},
        600: {"eta_sig": 0.5, "eta_stop": 0.1, "max_layers": None, "t": 10000},
    },
    # SECTION: HTRU2_2, two_stage, selection_seeds=(10, 54, 24)
    12: {
        200: {"eta_sig": 0.55, "eta_stop": 0.1, "t": 10000, "max_layers": None},
        400: {"eta_sig": 0.85, "eta_stop": 0.5, "t": 10000, "max_layers": None},
        600: {"eta_sig": 0.7, "eta_stop": 0.5, "t": 100000, "max_layers": None},
        800: {"eta_sig": 0.95, "eta_stop": 0.1, "t": 10000, "max_layers": None},
    },
    # SECTION: GFE2, two_stage, selection_seeds=(10, 54, 24)
    15: {
        100: {"eta_sig": 0.9, "eta_stop": 0.1, "max_layers": None, "t": 10000000},
        150: {"eta_sig": 0.6, "eta_stop": 0.1, "max_layers": None, "t": 100000000},
        200: {"eta_sig": 0.6, "eta_stop": 0.3, "max_layers": None, "t": 100000},
        250: {"eta_sig": 0.75, "eta_stop": 0.5, "max_layers": None, "t": 10000},
    },
}


# Every registered dataset section documents the selection protocol and seeds
# used to choose its learned hyperparameters.
_PMOLPP_PARAMETER_PROVENANCE = {
    1: {
        300: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        350: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        400: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        450: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
    },
    6: {
        200: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        400: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        600: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        800: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
    },
    7: {
        300: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        400: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        500: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        600: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
    },
    12: {
        200: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        400: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        600: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        800: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
    },
    15: {
        100: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        150: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        200: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
        250: {"selection_protocol": "two_stage_three_seed_mean", "selection_seeds": "10,54,24"},
    },
}


def _validate_parameter_provenance_registry():
    if set(_PMOLPP_PARAMETER_CONFIG) != set(_PMOLPP_PARAMETER_PROVENANCE):
        raise ValueError("PMOLPP parameter/provenance registry data_num drift")
    for data_num, parameters_by_tspc in _PMOLPP_PARAMETER_CONFIG.items():
        provenance_by_tspc = _PMOLPP_PARAMETER_PROVENANCE[data_num]
        if set(parameters_by_tspc) != set(provenance_by_tspc):
            raise ValueError(
                "PMOLPP parameter/provenance registry TSPC drift for "
                "data_num={}".format(data_num)
            )
        for tspc, provenance in provenance_by_tspc.items():
            if not isinstance(provenance, Mapping):
                raise ValueError(
                    "PMOLPP provenance must be a mapping for "
                    "data_num={}, tspc={}".format(data_num, tspc)
                )
            for name in ("selection_protocol", "selection_seeds"):
                value = provenance.get(name)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        "PMOLPP provenance {} must be non-empty text for "
                        "data_num={}, tspc={}".format(name, data_num, tspc)
                    )


def HasPMOLPPParameters(data_num, tspc):
    _validate_parameter_provenance_registry()
    for name,value in (("data_num",data_num),("tspc",tspc)):
        if isinstance(value,bool) or not isinstance(value,numbers.Integral):
            raise ValueError("{}必须是整数（不接受布尔值）。".format(name))
    return int(data_num) in _PMOLPP_PARAMETER_CONFIG and int(tspc) in _PMOLPP_PARAMETER_CONFIG[int(data_num)]


def Data_num2PMOLPP_Provenance(data_num, tspc):
    """Return a copy of the documented selection provenance for one entry."""
    _validate_parameter_provenance_registry()
    for name, value in (("data_num", data_num), ("tspc", tspc)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError("{}必须是整数（不接受布尔值）。".format(name))
    data_num = int(data_num)
    tspc = int(tspc)
    if data_num not in _PMOLPP_PARAMETER_PROVENANCE:
        raise ValueError(
            "未知 data_num={}，可用值为 {}".format(
                data_num, sorted(_PMOLPP_PARAMETER_PROVENANCE)
            )
        )
    if tspc not in _PMOLPP_PARAMETER_PROVENANCE[data_num]:
        available = sorted(_PMOLPP_PARAMETER_PROVENANCE[data_num])
        raise ValueError(
            "data_num={} 不支持 tspc={}，可用值为 {}".format(
                data_num, tspc, available
            )
        )
    return dict(_PMOLPP_PARAMETER_PROVENANCE[data_num][tspc])


def Data_num2PMOLPP_Parameters(data_num, tspc):
    """根据数据规模和 TSPC 返回学习得到的 PMOLPP 参数副本。"""
    _validate_parameter_provenance_registry()
    for name, value in (("data_num", data_num), ("tspc", tspc)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(
                f"{name} 必须是整数（不接受布尔值），收到 {value!r}"
            )

    data_num = int(data_num)
    tspc = int(tspc)
    if data_num not in _PMOLPP_PARAMETER_CONFIG:
        raise ValueError(f"未知 data_num={data_num}，可用值为 {sorted(_PMOLPP_PARAMETER_CONFIG)}")
    if tspc not in _PMOLPP_PARAMETER_CONFIG[data_num]:
        available = sorted(_PMOLPP_PARAMETER_CONFIG[data_num])
        raise ValueError(f"data_num={data_num} 不支持 tspc={tspc}，可用值为 {available}")
    return dict(_PMOLPP_PARAMETER_CONFIG[data_num][tspc])

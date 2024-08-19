"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 18:51:52
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : TOPSIS.py
# @Software: PyCharm
-------------------------------------
"""

import pandas as pd
import numpy as np


def entropy_weight(features):
    """

    Entropy method

    Args:
        features: Features

    Returns:
        weight: weight coefficient

    """
    features = np.array(features)
    proportion = features / features.sum(axis=0)  # normalized
    entropy = np.nansum(-proportion * np.log(proportion) / np.log(len(features)),
                        axis=0)  # calculate entropy
    weight = (1 - entropy) / (1 - entropy).sum()

    return weight  # calculation weight coefficient


def topsis(data, weight=None):
    """

    TOPSIS algorithm

    Args:
        data: Features
        weight:

    Returns:
        Result:
        Z:
        weight:

    """
    # normalized
    data = data / np.sqrt((data ** 2).sum())
    # best and worst solution
    Z = pd.DataFrame([data.min(), data.max()], index=['璐熺悊鎯宠В', '姝ｇ悊鎯宠В'])

    weight = entropy_weight(data) if weight is None else np.array(weight)  # distance
    Result = data.copy()
    Result['姝ｇ悊鎯宠В'] = np.sqrt(((data - Z.loc['姝ｇ悊鎯宠В']) ** 2 * weight).sum(axis=1))
    Result['璐熺悊鎯宠В'] = np.sqrt(((data - Z.loc['璐熺悊鎯宠В']) ** 2 * weight).sum(axis=1))

    # composite score index
    Result['缁煎悎寰楀垎鎸囨暟'] = Result['璐熺悊鎯宠В'] / (Result['璐熺悊鎯宠В'] + Result['姝ｇ悊鎯宠В'])
    Result['鎺掑簭'] = Result.rank(ascending=False)['缁煎悎寰楀垎鎸囨暟']

    return Result, Z, weight

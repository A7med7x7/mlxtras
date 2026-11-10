"""Preprocessing utilities for tabular data.

This module contains helpers for preparing pandas DataFrames before training
or evaluation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def reduce_memory_usage(
    df: pd.DataFrame,
    convert_object_to_category: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Reduce memory usage of a pandas DataFrame by downcasting numeric dtypes.

    The function inspects each column and converts integer and floating-point
    data to smaller NumPy dtypes when the range fits. Object columns can also
    be converted to categorical dtype to save memory.

    Parameters
    ----------
    df:
        Input DataFrame to process.
    convert_object_to_category:
        If True, convert object columns to ``category`` dtype.
    copy:
        If True, return a copy of the DataFrame. If False, modify the original
        DataFrame in place and return it.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with memory-efficient dtypes.
    """
    if copy:
        df = df.copy()

    for column in df.columns:
        column_dtype = str(df[column].dtype.name)

        if column_dtype.startswith("datetime") or column_dtype == "category":
            continue

        if column_dtype == "object" or column_dtype.startswith("str"):
            if convert_object_to_category:
                df[column] = df[column].astype("category")
            continue

        if column_dtype.startswith("int"):
            minimum = df[column].min()
            maximum = df[column].max()
            if minimum >= np.iinfo(np.int8).min and maximum <= np.iinfo(np.int8).max:
                df[column] = df[column].astype(np.int8)
            elif minimum >= np.iinfo(np.int16).min and maximum <= np.iinfo(np.int16).max:
                df[column] = df[column].astype(np.int16)
            elif minimum >= np.iinfo(np.int32).min and maximum <= np.iinfo(np.int32).max:
                df[column] = df[column].astype(np.int32)
            elif minimum >= np.iinfo(np.int64).min and maximum <= np.iinfo(np.int64).max:
                df[column] = df[column].astype(np.int64)
        elif column_dtype.startswith("float"):
            minimum = df[column].min()
            maximum = df[column].max()
            if minimum >= np.finfo(np.float16).min and maximum <= np.finfo(np.float16).max:
                df[column] = df[column].astype(np.float16)
            elif minimum >= np.finfo(np.float32).min and maximum <= np.finfo(np.float32).max:
                df[column] = df[column].astype(np.float32)

    return df

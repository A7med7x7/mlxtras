import pandas as pd
import numpy as np

from mlxtras.preprocessing import reduce_memory_usage


def test_reduce_memory_usage_converts_numeric_columns_to_smaller_dtypes():
    df = pd.DataFrame(
        {
            "small_int": [1, 2, 3],
            "large_int": [1, 2, 3000000000],
            "float_col": [1.1, 2.2, 3.3],
            "text_col": ["a", "b", "c"],
        }
    )

    result = reduce_memory_usage(df, convert_object_to_category=True)

    assert result["small_int"].dtype == np.int8
    assert result["float_col"].dtype == np.float16
    assert result["text_col"].dtype == "category"
    assert result["large_int"].dtype == np.int64


def test_reduce_memory_usage_preserves_original_dataframe_when_copy_false():
    df = pd.DataFrame({"value": [1, 2, 3]})

    result = reduce_memory_usage(df, copy=False)

    assert result is df
    assert result["value"].dtype == np.int8


def test_reduce_memory_usage_handles_booleans_and_non_converted_strings():
    df = pd.DataFrame(
        {
            "flag": [True, False, True],
            "text": ["foo", "bar", "baz"],
            "dates": pd.date_range("2026-01-01", periods=3),
        }
    )

    result = reduce_memory_usage(df, convert_object_to_category=False)

    assert result["flag"].dtype == bool
    assert result["dates"].dtype.name.startswith("datetime64")
    assert result["text"].dtype == "object" or str(result["text"].dtype).startswith("str")

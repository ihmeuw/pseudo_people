import numpy as np
from pandas._typing import DtypeObj as pd_dtype

from pseudopeople.dtypes import DtypeNames


def output_dtype_getter_leave_blank(dtype: pd_dtype) -> pd_dtype:
    # Make sure the dtype is nullable
    if "int" in dtype.name:
        return np.dtype("float")

    return dtype


def output_dtype_getter_always_string(_: pd_dtype) -> pd_dtype:
    return np.dtype(DtypeNames.OBJECT)

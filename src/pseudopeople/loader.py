from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.utilities import PANDAS_ENGINE, DataFrame, Engine


def load_standard_dataset(
    data_path: Path,
    user_filters: List[Tuple],
    engine: Engine = PANDAS_ENGINE,
    is_file: bool = True,
) -> DataFrame:
    if is_file and data_path.suffix != ".parquet":
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {data_path.suffix}"
        )

    if engine == PANDAS_ENGINE:
        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(data_path, filters=user_filters).to_pandas()
    else:
        # Modin
        import modin.pandas as mpd

        # NOTE: Modin doesn't work with PosixPath types
        # TODO: released versions of Modin can't actually distribute `filters`, see https://github.com/modin-project/modin/issues/5509
        # So released modin doesn't actually get us distributed loading of the data, and it all needs to fit into
        # memory on a single machine, which mostly beats the point.
        # This has been fixed in the master branch of Modin's GitHub, so you need to install from source:
        # pip install git+https://github.com/modin-project/modin.git
        data = mpd.read_parquet(str(data_path), filters=user_filters)

    if not isinstance(data, engine.dataframe_class):
        raise DataSourceError(
            f"File located at {data_path} must contain a DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )

    # TODO: The index in our simulated population files is never meaningful.
    # For some reason, the 1040 dataset is currently saved with a non-RangeIndex.
    # If we don't reset here, our index can have duplicates, which breaks much of
    # our noising logic.
    return data.reset_index(drop=True)

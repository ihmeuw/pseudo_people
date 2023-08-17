from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError

if TYPE_CHECKING:
    from pseudopeople.utilities import DATAFRAME, ENGINE


def load_standard_dataset_file(
    data_path: Path, user_filters: List[Tuple], engine: ENGINE = "pandas"
) -> DATAFRAME:
    if engine == "pandas":
        if data_path.suffix != ".parquet":
            raise DataSourceError(
                f"Source path must be a .parquet file. Provided {data_path.suffix}"
            )

        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(data_path, filters=user_filters).to_pandas()

        if not isinstance(data, pd.DataFrame):
            raise DataSourceError(
                f"File located at {data_path} must contain a pandas DataFrame. "
                "Please provide the path to the unmodified root data directory."
            )

        return data
    else:
        # Modin
        import modin.pandas as mpd

        # NOTE: Modin doesn't work with PosixPath types
        # TODO: released versions of Modin can't actually distribute `filters`, see https://github.com/modin-project/modin/issues/5509
        # So for now, modin doesn't actually get us distributed loading of the data, and it all needs to fit into
        # memory on a single machine, which mostly beats the point.
        # This has been fixed in the master branch of Modin's GitHub, but we can't use a bleeding edge version
        # because it requires pandas>=2.0.0 which Vivarium doesn't support yet.
        # For now, install modin from the modin_22_backport_parquet_filters branch at https://github.com/zmbc/modin
        return mpd.read_parquet(str(data_path), filters=user_filters)

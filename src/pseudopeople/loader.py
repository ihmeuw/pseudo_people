# mypy: disable-error-code="unused-ignore"

from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from pseudopeople.exceptions import DataSourceError
from pseudopeople.filter import DataFilter
from pseudopeople.utilities import PANDAS_ENGINE, DataFrame, Engine


def load_standard_dataset(
    data_path: Path,
    filters: Sequence[DataFilter],
    engine: Engine = PANDAS_ENGINE,
    is_file: bool = True,
) -> DataFrame:
    if is_file and data_path.suffix != ".parquet":
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {data_path.suffix}"
        )

    parquet_filters = [filter.to_tuple() for filter in filters] if filters else None
    if engine == PANDAS_ENGINE:
        if not parquet_filters:
            # pyarrow.parquet.read_table doesn't accept an empty list
            parquet_filters = None
        data: DataFrame = pq.read_table(
            str(data_path),
            filters=parquet_filters,  # type: ignore [arg-type]
        ).to_pandas()
    else:
        # Dask
        import dask.dataframe as dd

        data = dd.read_parquet(str(data_path), filters=parquet_filters)

    # TODO: The index in our simulated population files is never meaningful.
    # For some reason, the 1040 dataset is currently saved with a non-RangeIndex
    # in the large data, and all datasets have a non-RangeIndex in the sample data.
    # If we don't drop these here, our index can have duplicates when we load multiple
    # shards at once. Having duplicates in the index breaks much of
    # our noising logic.
    data = data.reset_index(drop=True)

    if not isinstance(data, engine.dataframe_class):
        raise DataSourceError(
            f"File located at {data_path} must contain a DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )

    return data

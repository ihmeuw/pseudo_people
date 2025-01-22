from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from pseudopeople.configuration import Keys
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.metadata import DATEFORMATS
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.dtypes import DtypeNames
from pseudopeople.entity_types import ColumnNoiseType, NoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DatasetSchema
from pseudopeople.utilities import coerce_dtypes, get_random_generator, to_string


def noise_data(
    dataset_schema: DatasetSchema,
    data: pd.DataFrame,
    seed: Any,
    configuration: NoiseConfiguration,
    progress_bar: bool = True,
) -> pd.DataFrame:
    return Dataset(dataset_schema, data, seed).get_noised_data(
        configuration, noise_types=NOISE_TYPES, progress_bar=progress_bar
    )


class Dataset:
    def __init__(
        self,
        dataset_schema: DatasetSchema,
        data: pd.DataFrame,
        seed: Any,
    ):
        self.dataset_schema = dataset_schema
        self.data = data
        self.randomness = get_random_generator(self.dataset_schema.name, seed)
        self.missingness = self.is_missing(self.data)

    def __bool__(self) -> bool:
        return not self.data.empty

    def is_empty(self, column_name: str) -> Any:
        """Returns whether the column is empty."""
        is_empty: bool = self.missingness[column_name].all()
        return is_empty

    def get_non_empty_index(self, required_columns: list[str] | None = None) -> pd.Index[int]:
        """Returns the non-empty data."""

        if required_columns is None:
            non_empty_data = self.data.loc[~self.missingness.all(axis=1)]
        else:
            missingness_mask = self.missingness[required_columns].any(axis=1)
            non_empty_data = self.data.loc[~missingness_mask, required_columns]
        return non_empty_data.index

    def get_noised_data(
        self,
        configuration: NoiseConfiguration,
        noise_types: Sequence[NoiseType],
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Returns the noised dataset data."""
        self._clean_input_data()
        self._reformat_dates_for_noising()
        self._noise_dataset(configuration, noise_types, progress_bar=progress_bar)
        self.data = coerce_dtypes(self.data, self.dataset_schema)
        self.data = self.data[[c.name for c in self.dataset_schema.columns]]
        return self.data

    def _noise_dataset(
        self,
        configuration: NoiseConfiguration,
        noise_types: Sequence[NoiseType],
        progress_bar: bool = True,
    ) -> None:
        """
        Adds noise to the dataset data. Noise functions are executed in the order
        defined by :py:const: `.NOISE_TYPES`. Row noise functions are applied to the
        whole DataFrame. Column noise functions will be applied to each column that
        is pertinent to it.

        Noise levels are determined by the noise_config.
        :param configuration:
            Object to configure noise levels
        """
        if progress_bar:
            noise_type_iterator: Sequence[NoiseType] | tqdm[NoiseType] = tqdm(
                noise_types, desc="Applying noise", unit="type", leave=False
            )
        else:
            noise_type_iterator = noise_types

        for noise_type in noise_type_iterator:
            if isinstance(noise_type, RowNoiseType):
                if configuration.has_noise_type(self.dataset_schema.name, noise_type.name):
                    # Apply row noise
                    noise_type(self, configuration)

            elif isinstance(noise_type, ColumnNoiseType):
                for column in self.data.columns:
                    if configuration.has_noise_type(
                        self.dataset_schema.name, noise_type.name, column
                    ):
                        noise_type(self, configuration, column)

            else:
                raise TypeError(
                    f"Invalid noise type. Allowed types are {RowNoiseType} and "
                    f"{ColumnNoiseType}. Provided {type(noise_type)}."
                )

    def _clean_input_data(self) -> None:
        for col in self.dataset_schema.columns:
            # Coerce empty strings to nans
            self.data[col.name] = self.data[col.name].replace("", np.nan)

            if (
                self.data[col.name].dtype.name == "category"
                and col.dtype_name == DtypeNames.OBJECT
            ):
                # We made some columns in the pseudopeople input categorical
                # purely as a kind of DIY compression.
                # TODO: Determine whether this is benefitting us after
                # the switch to Parquet.
                self.data[col.name] = to_string(self.data[col.name])

    def _reformat_dates_for_noising(self) -> None:
        """Formats date columns so they can be noised as strings."""
        data = self.data.copy()

        for date_column in [COLUMNS.dob.name, COLUMNS.ssa_event_date.name]:
            # Format both the actual column, and the shadow version that will be used
            # to copy from a household member
            for column in [date_column, COPY_HOUSEHOLD_MEMBER_COLS.get(date_column)]:
                if column in data.columns and isinstance(column, str):
                    # Avoid running strftime on large data, since that will
                    # re-parse the format string for each row
                    # https://github.com/pandas-dev/pandas/issues/44764
                    # Year is already guaranteed to be 4-digit: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-timestamp-limits
                    is_na = data[column].isna()
                    try:
                        data_column = pd.to_datetime(data.loc[~is_na, column])
                    except:
                        breakpoint()
                    year_string = data_column.dt.year.astype(str)
                    month_string = _zfill_fast(data_column.dt.month.astype(str), 2)
                    day_string = _zfill_fast(data_column.dt.day.astype(str), 2)
                    if self.dataset_schema.date_format == DATEFORMATS.YYYYMMDD:
                        result = year_string + month_string + day_string
                    elif self.dataset_schema.date_format == DATEFORMATS.MM_DD_YYYY:
                        result = month_string + "/" + day_string + "/" + year_string
                    elif self.dataset_schema.date_format == DATEFORMATS.MMDDYYYY:
                        result = month_string + day_string + year_string
                    else:
                        raise ValueError(
                            f"Invalid date format in {self.dataset_schema.name}."
                        )

                    data[column] = pd.Series(np.nan, dtype=str)
                    data.loc[~is_na, column] = result

        self.data = data

    @staticmethod
    def is_missing(data: pd.DataFrame) -> pd.DataFrame:
        """Returns a boolean dataframe with the same columns, index, and shape of
        the data attribute. Boolean dataframe is True if a cell is missing, False otherwise.
        """
        return (data == "") | (data.isna())


def _zfill_fast(col: pd.Series[str], desired_length: int) -> pd.Series[str]:
    """Performs the same operation as col.str.zfill(desired_length), but vectorized."""
    # The most zeroes that could ever be needed would be desired_length
    maximum_padding = ("0" * desired_length) + col
    # Now trim to only the zeroes needed
    return maximum_padding.str[-desired_length:]

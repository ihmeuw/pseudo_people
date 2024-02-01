from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from vivarium import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.constants.metadata import DATEFORMATS
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import Dataset, COLUMNS
from pseudopeople.utilities import get_randomness_stream, coerce_dtypes

from pseudopeople.loader import load_standard_dataset_file


class DatasetData:
    def __init__(self, dataset: Dataset, data_path: Path, user_filters: List[Tuple], seed: str):
        self.dataset = dataset
        self.randomness = get_randomness_stream(self.dataset.name, seed, self.data.index)
        self.data = load_standard_dataset_file(data_path, user_filters)
        self.missingness = self.is_missing(self.data)

    def __bool__(self):
        return self.data.empty

    def is_empty(self, column_name: str) -> bool:
        """Returns whether the column is empty."""
        return self.missingness[column_name].all()

    def get_non_empty_index(self, required_columns: Optional[List[str]] = None) -> pd.Index:
        """Returns the non-empty data."""
        if required_columns is None:
            non_empty_data = self.data.loc[~self.missingness.all(axis=1)]
        else:
            missingness_mask = self.missingness[required_columns].any(axis=1)
            non_empty_data = self.data.loc[~missingness_mask, required_columns]
        return non_empty_data.index

    def get_noised_data(self, configuration: ConfigTree) -> pd.DataFrame:
        """ Returns the noised dataset data. """
        self.format_data()
        self.noise_dataset(configuration)
        self.drop_extra_columns()
        return self.data

    def format_data(self) -> None:
        """Formats the data to match the expected format for noising."""
        self._reformat_dates_for_noising()
        self.data = coerce_dtypes(self.data, self.dataset)

    def noise_dataset(self, configuration: ConfigTree) -> None:
        """
        Adds noise to the dataset data. Noise functions are executed in the order
        defined by :py:const: `.NOISE_TYPES`. Row noise functions are applied to the
        whole DataFrame. Column noise functions will be applied to each column that
        is pertinent to it.

        Noise levels are determined by the noise_config.
        :param configuration:
            Object to configure noise levels
        """

        noise_configuration = configuration[self.dataset.name]

        for noise_type in tqdm(NOISE_TYPES, desc="Applying noise", unit="type", leave=False):
            if isinstance(noise_type, RowNoiseType):
                if (
                    Keys.ROW_NOISE in noise_configuration
                    and noise_type.name in noise_configuration.row_noise
                ):
                    # Apply row noise
                    noise_type(self, noise_configuration[Keys.ROW_NOISE][noise_type.name])

                    # todo this logic needs to move inside the duplicate row noise function
                    # Check for duplicated rows
                    # new_indices = self.data.index.difference(original_index)
                    # new_missingness = self.data.loc[new_indices].isna() | (
                    #     self.data.loc[new_indices] == ""
                    # )
                    # missingness = pd.concat([missingness, new_missingness])

            elif isinstance(noise_type, ColumnNoiseType):
                if Keys.COLUMN_NOISE in noise_configuration:
                    columns_to_noise = [
                        col
                        for col in noise_configuration.column_noise
                        if col in self.data.columns
                        and noise_type.name in noise_configuration.column_noise[col]
                    ]
                    # Apply column noise to each column as appropriate
                    for column in columns_to_noise:
                        noise_type(
                            self,
                            noise_configuration.column_noise[column][noise_type.name],
                            column,
                        )

                        # todo this logic needs to move inside the leave blank noise function
                        # if noise_type == NOISE_TYPES.leave_blank:
                        #     # The only situation in which more missingness is introduced
                        #     missingness.loc[index_noised, column] = True
            else:
                raise TypeError(
                    f"Invalid noise type. Allowed types are {RowNoiseType} and "
                    f"{ColumnNoiseType}. Provided {type(noise_type)}."
                )

    def _reformat_dates_for_noising(self) -> None:
        """Formats date columns so they can be noised as strings."""
        data = self.data.copy()

        for date_column in [COLUMNS.dob.name, COLUMNS.ssa_event_date.name]:
            # Format both the actual column, and the shadow version that will be used
            # to copy from a household member
            for column in [date_column, COPY_HOUSEHOLD_MEMBER_COLS.get(date_column)]:
                if column in data.columns:
                    # Avoid running strftime on large data, since that will
                    # re-parse the format string for each row
                    # https://github.com/pandas-dev/pandas/issues/44764
                    # Year is already guaranteed to be 4-digit: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-timestamp-limits
                    is_na = data[column].isna()
                    data_column = data.loc[~is_na, column]
                    year_string = data_column.dt.year.astype(str)
                    month_string = _zfill_fast(data_column.dt.month.astype(str), 2)
                    day_string = _zfill_fast(data_column.dt.day.astype(str), 2)
                    if self.dataset.date_format == DATEFORMATS.YYYYMMDD:
                        result = year_string + month_string + day_string
                    elif self.dataset.date_format == DATEFORMATS.MM_DD_YYYY:
                        result = month_string + "/" + day_string + "/" + year_string
                    elif self.dataset.date_format == DATEFORMATS.MMDDYYYY:
                        result = month_string + day_string + year_string
                    else:
                        raise ValueError(f"Invalid date format in {self.dataset.name}.")

                    data[column] = pd.Series(np.nan, dtype=str)
                    data.loc[~is_na, column] = result

        self.data = data


    def drop_extra_columns(self) -> None:
        """Drops columns that are not in the dataset schema."""
        self.data = self.data[[c.name for c in self.dataset.columns]]

    @staticmethod
    def is_missing(data: pd.DataFrame) -> pd.DataFrame:
        """Returns the missingness of the data."""
        return (data == "") | (data.isna())


def _zfill_fast(col: pd.Series, desired_length: int) -> pd.Series:
    """Performs the same operation as col.str.zfill(desired_length), but vectorized."""
    # The most zeroes that could ever be needed would be desired_length
    maximum_padding = ("0" * desired_length) + col
    # Now trim to only the zeroes needed
    return maximum_padding.str[-desired_length:]

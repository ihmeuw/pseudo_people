from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.configuration import Keys
from pseudopeople.utilities import get_index_to_noise


@dataclass
class RowNoiseType:
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise type (e.g. "omit_row" or
    "duplicate_row").

    The noise function takes as input a DataFrame, the configuration value
    for this RowNoise operation, and a RandomnessStream for controlling
    randomness. It applies the noising operation to the entire DataFrame and
    returns the modified DataFrame.
    """

    name: str
    noise_function: Callable[[str, pd.DataFrame, ConfigTree, RandomnessStream], None]
    row_probability: float = 0.0

    def __call__(
        self,
        dataset_name: str,
        dataset_data: pd.DataFrame,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
    ) -> None:
        self.noise_function(dataset_name, dataset_data, configuration, randomness_stream)


@dataclass
class ColumnNoiseType:
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise type (e.g. use_nickname" or
    "make_phonetic_errors").

    The noise function takes as input a DataFrame, the ConfigTree object for this
    ColumnNoise operation, a RandomnessStream for controlling randomness, and
    a column name, which is the column that will be noised and who's name will be used
    as the additional key for the RandomnessStream. It applies the noising operation
    to the Series and returns the modified Series.
    """

    name: str
    noise_function: Callable[
        [pd.DataFrame, pd.Index, ConfigTree, RandomnessStream, str, str], None
    ]
    cell_probability: Optional[float] = 0.01
    noise_level_scaling_function: Callable[[pd.DataFrame, str], float] = lambda x, y: 1.0
    additional_parameters: Dict[str, Any] = None
    additional_column_getter: Callable[[str], List[str]] = lambda column_name: []

    def __call__(
        self,
        data: pd.DataFrame,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
        dataset_name: str,
        column_name: str,
        required_cols: Optional[List[str]] = None,
    ) -> None:
        if data[column_name].empty:
            return

        noise_level = configuration[
            Keys.CELL_PROBABILITY
        ] * self.noise_level_scaling_function(data, column_name)
        # Certain columns have their noise level scaled so we must check to make sure the noise level is within the
        # allowed range between 0 and 1 for probabilities
        noise_level = min(noise_level, 1.0)
        if required_cols is None:
            required_cols = [column_name]
        to_noise_idx = get_index_to_noise(
            data,
            noise_level,
            randomness_stream,
            f"{self.name}_{column_name}",
            ignore_rows_missing_columns=required_cols,
        )
        if to_noise_idx.empty:
            logger.debug(
                f"No cells chosen to noise for noise function {self.name} on column {column_name}. "
                "This is likely due to a combination of the configuration noise levels and the simulated population data."
            )
            return

        dtype_before = data[column_name].dtype
        self.noise_function(
            data,
            to_noise_idx,
            configuration,
            randomness_stream,
            dataset_name,
            column_name,
        )

        # Coerce noised column dtype back to original column's if it has changed
        if data[column_name].dtype.name != dtype_before.name:
            data[column_name] = data[column_name].astype(dtype_before)

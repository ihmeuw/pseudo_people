from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from layered_config_tree import LayeredConfigTree
from loguru import logger
from pandas._typing import DtypeObj as pd_dtype

from pseudopeople.configuration import Keys
from pseudopeople.utilities import ensure_dtype, get_index_to_noise

if TYPE_CHECKING:
    from pseudopeople.configuration.noise_configuration import NoiseConfiguration


def _noise_function_not_implemented(*_args: Any, **_kwargs: Any) -> None:
    pass


def default_noise_level_getter(
    configuration: NoiseConfiguration, dataset: Dataset, noise_type: str
) -> float:
    noise_level: float = configuration.get_row_probability(
        dataset.dataset_schema.name, noise_type
    )
    return noise_level


if TYPE_CHECKING:
    from pseudopeople.dataset import Dataset


@dataclass
class NoiseType(ABC):
    name: str
    noise_function: Callable[..., None] = _noise_function_not_implemented
    probability: float | None = 0.0
    additional_parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.noise_function == _noise_function_not_implemented:
            raise NotImplementedError(
                "You must pass a noise_function when creating a NoiseType. "
                f"No noise_function provided to NoiseType {self.name}."
            )

    @property
    @abstractmethod
    def probability_key(self) -> str:
        pass


@dataclass
class RowNoiseType(NoiseType):
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise type (e.g. "omit_row" or
    "duplicate_row").

    The noise function takes as input a DataFrame, the configuration value
    for this RowNoise operation, and a np.random.default_rng for controlling
    randomness. It applies the noising operation to the entire DataFrame and
    returns the modified DataFrame.
    """

    noise_function: Callable[
        [Dataset, NoiseConfiguration, pd.Index[int]], None
    ] = _noise_function_not_implemented
    get_noise_level: Callable[
        [NoiseConfiguration, Dataset, str], float | pd.Series[int | float]
    ] = default_noise_level_getter

    @property
    def probability_key(self) -> str:
        return Keys.ROW_PROBABILITY

    def __call__(self, dataset: Dataset, configuration: NoiseConfiguration) -> None:
        noise_level = self.get_noise_level(configuration, dataset, self.name)
        to_noise_idx = get_index_to_noise(dataset, noise_level)
        self.noise_function(dataset, configuration, to_noise_idx)


@dataclass
class ColumnNoiseType(NoiseType):
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise type (e.g. use_nickname" or
    "make_phonetic_errors").

    The noise function takes as input a DataFrame, the NoiseConfiguration object for this
    ColumnNoise operation, a np.random.default_rng for controlling randomness.
    Optionally, it can take a pre-existing DataFrame indicating where there is missingness
    in the data (same index and columns as the main DataFrame, all boolean type) --
    if this is not passed, it calculates it, which can be expensive for large data.
    It applies the noising operation to the Series and returns both the modified Series
    and an Index of which items in the Series were selected for noise.
    """

    noise_function: Callable[
        [Dataset, NoiseConfiguration, pd.Index[int], str], None
    ] = _noise_function_not_implemented
    probability: float | None = 0.01
    noise_level_scaling_function: Callable[[pd.DataFrame, str], float] = lambda x, y: 1.0
    additional_column_getter: Callable[[str], list[str]] = lambda column_name: []
    output_dtype_getter: Callable[[pd_dtype], pd_dtype] = lambda dtype: dtype

    @property
    def probability_key(self) -> str:
        return Keys.CELL_PROBABILITY

    def __call__(
        self,
        dataset: Dataset,
        configuration: NoiseConfiguration,
        column_name: str,
    ) -> None:
        if dataset.is_empty(column_name):
            return
        cell_probability: float = configuration.get_cell_probability(
            dataset.dataset_schema.name, self.name, column_name
        )
        noise_level = cell_probability * self.noise_level_scaling_function(
            dataset.data, column_name
        )

        # Certain columns have their noise level scaled so we must check to make
        # sure the noise level is within the allowed range between 0 and 1 for
        # probabilities
        noise_level = min(noise_level, 1.0)
        to_noise_idx = get_index_to_noise(
            dataset,
            noise_level,
            [column_name] + self.additional_column_getter(column_name),
        )
        if to_noise_idx.empty:
            logger.debug(
                f"No cells chosen to noise for noise function {self.name} on "
                f"column {column_name}. This is likely due to a combination "
                "of the configuration noise levels and the simulated population data."
            )
            return

        input_dtype = dataset.data[column_name].dtype
        output_dtype = self.output_dtype_getter(input_dtype)

        dataset.data[column_name] = ensure_dtype(dataset.data[column_name], output_dtype)

        self.noise_function(
            dataset,
            configuration,
            to_noise_idx,
            column_name,
        )

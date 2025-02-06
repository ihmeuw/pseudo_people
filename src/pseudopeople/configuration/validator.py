from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from layered_config_tree import ConfigurationKeyError, LayeredConfigTree
from loguru import logger

from pseudopeople.configuration import Keys
from pseudopeople.constants import metadata, paths
from pseudopeople.exceptions import ConfigurationError
from pseudopeople.filter import DataFilter
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_scaling import get_options_for_column
from pseudopeople.schema_entities import DatasetSchema


class ParameterConfigValidator(Protocol):
    def __call__(
        self, noise_type_config: Any, parameter: str, base_error_message: str, **_: Any
    ) -> None:
        ...


def validate_overrides(overrides: Any, default_config: LayeredConfigTree) -> None:
    """
    Validates the user-provided overrides. Confirms that all user-provided
    keys exist in the default configuration. Confirms that all user-provided
    values are valid for their respective noise functions.
    """
    if not isinstance(overrides, dict):
        raise ConfigurationError("Invalid configuration type provided.") from None
    for dataset_name, dataset_config in overrides.items():
        if not isinstance(dataset_config, dict):
            raise ConfigurationError(
                f"'{dataset_name}' must be a Dict. "
                f"Provided {dataset_config} of type {type(dataset_config)}."
            )

        default_dataset_config = _get_default_config_node(
            default_config, dataset_name, "dataset"
        )
        for key in dataset_config:
            _get_default_config_node(
                default_dataset_config, key, "configuration key", dataset_name
            )

        default_row_noise_config: LayeredConfigTree = default_dataset_config.get_tree(
            Keys.ROW_NOISE
        )
        default_column_noise_config: LayeredConfigTree = default_dataset_config.get_tree(
            Keys.COLUMN_NOISE
        )

        row_noise_config = dataset_config.get(Keys.ROW_NOISE, {})
        if not isinstance(row_noise_config, dict):
            raise ConfigurationError(
                f"'{Keys.ROW_NOISE}' of '{dataset_name}' must be a Dict. "
                f"Provided {row_noise_config} of type {type(row_noise_config)}."
            )

        for noise_type, noise_type_config in row_noise_config.items():
            if not isinstance(noise_type_config, dict):
                raise ConfigurationError(
                    f"Row noise type '{noise_type}' of dataset '{dataset_name}' must be a Dict. "
                    f"Provided {noise_type_config} of type {type(noise_type_config)}."
                )
            default_noise_type_config = _get_default_config_node(
                default_row_noise_config, noise_type, "noise type", dataset_name
            )
            _validate_noise_type_config(
                noise_type_config,
                default_noise_type_config,
                dataset_name,
                noise_type,
                DEFAULT_PARAMETER_CONFIG_VALIDATOR_MAP,
            )

        column_noise_config = dataset_config.get(Keys.COLUMN_NOISE, {})
        if not isinstance(column_noise_config, dict):
            raise ConfigurationError(
                f"'{Keys.COLUMN_NOISE}' of '{dataset_name}' must be a Dict. "
                f"Provided {column_noise_config} of type {type(column_noise_config)}."
            )

        for column, column_config in column_noise_config.items():
            if not isinstance(column_config, dict):
                raise ConfigurationError(
                    f"Column '{column}' of dataset '{dataset_name}' must be a Dict. "
                    f"Provided {column_config} of type {type(column_config)}."
                )

            default_column_config = _get_default_config_node(
                default_column_noise_config, column, "column", dataset_name
            )
            for noise_type, noise_type_config in column_config.items():
                if not isinstance(noise_type_config, dict):
                    raise ConfigurationError(
                        f"Noise type '{noise_type}' of column '{column}' in dataset '{dataset_name}' must be a Dict. "
                        f"Provided {noise_type_config} of type {type(noise_type_config)}."
                    )

                default_noise_type_config = _get_default_config_node(
                    default_column_config, noise_type, "noise type", dataset_name, column
                )
                parameter_config_validator_map: dict[str, ParameterConfigValidator] = {
                    NOISE_TYPES.choose_wrong_option.name: {
                        Keys.CELL_PROBABILITY: _validate_choose_wrong_option_probability
                    },
                }.get(noise_type, DEFAULT_PARAMETER_CONFIG_VALIDATOR_MAP)
                _validate_noise_type_config(
                    noise_type_config,
                    default_noise_type_config,
                    dataset_name,
                    noise_type,
                    parameter_config_validator_map,
                    column,
                )


def _validate_noise_type_config(
    noise_type_config: dict[str, Any],
    default_noise_type_config: LayeredConfigTree,
    dataset_name: str,
    noise_type: str,
    parameter_config_validator_map: dict[str, ParameterConfigValidator],
    column: str | None = None,
) -> None:
    """
    Validates that all parameters are allowed for this noise function.
    Additionally, validates that the configuration values are permissible.
    """
    for parameter, parameter_config in noise_type_config.items():
        parameter_config_validator = parameter_config_validator_map.get(
            parameter, _validate_probability
        )

        _ = _get_default_config_node(
            default_noise_type_config,
            parameter,
            "parameter",
            dataset_name,
            column,
            noise_type,
        )
        base_error_message = (
            f"Invalid '{parameter}' provided for dataset '{dataset_name}' for "
            f"column '{column}' and noise type '{noise_type}'. "
        )
        parameter_config_validator(
            parameter_config, parameter, base_error_message, column=column
        )


def _get_default_config_node(
    default_config: LayeredConfigTree,
    key: str,
    key_type: str,
    dataset_name: str | None = None,
    column: str | None = None,
    noise_type: str | None = None,
) -> LayeredConfigTree:
    """
    Validate that the node the user is trying to add exists in the default
    configuration.
    """
    config_node: LayeredConfigTree = default_config.get(key)
    if config_node is None:
        dataset_context = "" if dataset_name is None else f" for dataset '{dataset_name}'"
        column_context = "" if column is None else f" for column '{column}'"
        noise_type_context = "" if noise_type is None else f" and noise type '{noise_type}'"
        context = dataset_context + column_context + noise_type_context

        error_message = f"Invalid {key_type} '{key}' provided{context}. "
        valid_options_message = f"Valid {key_type}s are {[k for k in default_config]}."
        raise ConfigurationError(error_message + valid_options_message) from None
    else:
        return config_node


def _validate_possible_age_differences(
    noise_type_config: Any,
    parameter: str,
    base_error_message: str,
    **_: Any,
) -> None:
    """
    Validates the user-provided values for the age-miswriting permutations
    parameter
    """
    if not isinstance(noise_type_config, (dict, list)):
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must be a Dict or List. "
            f"Provided {noise_type_config} of type {type(noise_type_config)}."
        )
    if len(noise_type_config) == 0:
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must not be empty. "
            f"Provided {noise_type_config}."
        )
    for key in noise_type_config:
        if not isinstance(key, int):
            raise ConfigurationError(
                base_error_message + f"'{parameter}' must be a List of ints. "
                f"Provided {key} of type {type(key)}."
            )
        if key == 0:
            raise ConfigurationError(base_error_message + f"'{parameter}' cannot include 0.")
    if isinstance(noise_type_config, dict):
        for value in noise_type_config.values():
            if not isinstance(value, (float, int)):
                raise ConfigurationError(
                    base_error_message
                    + f"'{parameter}' probabilities must be floats or ints. "
                    f"Provided {value} of type {type(value)}."
                )
            if not (0 <= value <= 1):
                raise ConfigurationError(
                    base_error_message
                    + f"'{parameter}' probabilities must be between 0 and 1 (inclusive). "
                    f"Provided {value} in {list(noise_type_config.values())}."
                )
        if not np.isclose(sum(noise_type_config.values()), 1.0):
            raise ConfigurationError(
                base_error_message + f"'{parameter}' probabilities must sum to 1. "
                f"Provided values sum to {sum(noise_type_config.values())}."
            )


def _validate_zipcode_digit_probabilities(
    noise_type_config: Any, parameter: str, base_error_message: str, **_: Any
) -> None:
    """Validates the user-provided values for the zipcode digit noising probabilities"""
    if not isinstance(noise_type_config, list):
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must be a List. "
            f"Provided {noise_type_config} of type {type(noise_type_config)}."
        )
    if len(noise_type_config) != 5:
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must be a List of 5 probabilities. "
            f"{len(noise_type_config)} probabilities ({noise_type_config})."
        )
    for value in noise_type_config:
        _validate_probability(value, parameter, base_error_message)


def _validate_probability(
    noise_type_config: Any, parameter: str, base_error_message: str, **_: Any
) -> None:
    if not isinstance(noise_type_config, (float, int)):
        raise ConfigurationError(
            base_error_message + f"'{parameter}' probabilities must be floats or ints. "
            f"Provided {noise_type_config} of type {type(noise_type_config)}."
        )

    if not (0 <= noise_type_config <= 1):
        raise ConfigurationError(
            base_error_message + f"'{parameter}'s must be between 0 and 1 (inclusive). "
            f"Provided {noise_type_config}."
        )


def _validate_choose_wrong_option_probability(
    noise_type_config: Any, parameter: str, base_error_message: str, **kwargs: Any
) -> None:
    column = kwargs.get("column")
    if not column:
        raise ValueError(
            "You must pass in a column argument when validating choose wrong option probabilities."
        )
    elif not isinstance(column, str):
        raise ConfigurationError("All the column names in your config must be strings.")

    _validate_probability(noise_type_config, parameter, base_error_message)
    num_options = len(get_options_for_column(column))
    # The maximum: if the cell *selection* probability were set to 1, and every cell
    # selected an option uniformly at random, how many cells would actually change?
    # Each cell would have a 1 / num_options chance of staying the same.
    maximum_noise_probability = 1 - (1 / num_options)
    if noise_type_config > maximum_noise_probability:
        logger.warning(
            base_error_message
            + f"The configured '{parameter}' is {noise_type_config}, but pseudopeople "
            f"can only choose the wrong option with a maximum of {maximum_noise_probability:.5f} probability. "
            f"This maximum will be used instead of the configured value."
        )


def validate_noise_level_proportions(
    configuration_tree: LayeredConfigTree,
    dataset_schema: DatasetSchema,
    filters: Sequence[DataFilter],
) -> None:
    """
    Validates that the noise levels provided do not exceed the allowable proportions from the
    metadata proportions file. If the provided noise levels are higher than the allowable proportions
    then throw a warning to the user and adjust the noise level to the allowable proportion.
    """
    # TODO: update file and filepath
    metadata_proportions = pd.read_csv(paths.METADATA_PROPORTIONS)
    dataset_proportions = metadata_proportions.loc[
        metadata_proportions["dataset"] == dataset_schema.name
    ]
    # Set default values for state and year
    if dataset_schema.name == metadata.DatasetNames.SSA:
        state = "USA"
    else:
        # Note: This is a shortcoming of our current approach to user warnings and will be fixed
        # with a future release/our next data upload. We do not have a way to get state in a
        # case where the user is not filtering on state because they either are using the sample
        # data or state (Rhode Island) data.
        if len(dataset_proportions["state"].unique()) == 1:
            state = dataset_proportions["state"].unique()[0]
        else:
            state = "USA"
    year = metadata.YEAR_AGGREGATION_VALUE
    # Get the state and year from the user filters
    for state_filter in filters:
        if state_filter.column_name == dataset_schema.state_column_name:
            state = str(state_filter.value)
            break
    for date_filter in filters:
        if date_filter.column_name == dataset_schema.date_column_name:
            year = (
                date_filter.value.year
                if isinstance(date_filter.value, pd.Timestamp)
                else int(date_filter.value)
            )
            break

    # Subset the metadata proportions to the state and year that the user is querying
    dataset_noise_proportions = dataset_proportions.loc[
        (dataset_proportions["state"] == state) & (dataset_proportions["year"] == year)
    ]

    # If there is no data for a queried dataset, we want the user's to hit the correct error that there
    # is no data available so we do not throw an error here.
    if not dataset_noise_proportions.empty:
        # Go through each row in the queried dataset noise proportions to validate the noise levels
        for i in range(len(dataset_noise_proportions)):
            row = dataset_noise_proportions.iloc[i].copy()
            column_names = [col.name for col in dataset_schema.columns]
            if row["column"] not in column_names and not pd.isnull(row["column"]):
                continue
            # Get the maximum noise level and the configured noise level
            if pd.isnull(row["column"]):
                # Note: Using pd.isnull here and above because np.isnan does not work on strings
                if NOISE_TYPES.duplicate_with_guardian in dataset_schema.row_noise_types:
                    # Config level for guardian duplication group
                    config_noise_level = (
                        configuration_tree.get_tree(row["dataset"])
                        .get_tree(Keys.ROW_NOISE)
                        .get_tree(NOISE_TYPES.duplicate_with_guardian.name)
                        .get(row["noise_type"])
                    )
                    entity_type = Keys.ROW_NOISE
                else:
                    # I have preloaded the metadata for ACS and CPS to have the duplicate with
                    # guardian metadata but we are not using it right now.
                    continue
            else:
                # Config level for each column noise type
                config_noise_level = (
                    configuration_tree.get_tree(row["dataset"])
                    .get_tree(Keys.COLUMN_NOISE)
                    .get_tree(row["column"])
                    .get_tree(row["noise_type"])
                    .get(Keys.CELL_PROBABILITY)
                )
                entity_type = Keys.COLUMN_NOISE
            max_noise_level = row["proportion"]
            if config_noise_level > max_noise_level:
                logger.warning(
                    f"The configured '{row['noise_type']}' noise level for {entity_type} '{row['column']}' is {config_noise_level}, "
                    f"which is higher than the maximum possible value based on the provided data for '{row['dataset']}'. "
                    "Noising as many rows as possible. "
                )


DEFAULT_PARAMETER_CONFIG_VALIDATOR_MAP: dict[str, ParameterConfigValidator] = {
    Keys.POSSIBLE_AGE_DIFFERENCES: _validate_possible_age_differences,
    Keys.ZIPCODE_DIGIT_PROBABILITIES: _validate_zipcode_digit_probabilities,
}

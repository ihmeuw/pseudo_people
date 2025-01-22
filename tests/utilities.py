from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytest_check import check
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants import paths
from pseudopeople.dataset import Dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS, Column
from tests.constants import CELL_PROBABILITY, TOKENS_PER_STRING_MAPPER
from tests.unit.test_configuration import COLUMN_NOISE_TYPES


def run_column_noising_tests(
    dataset_name: str,
    config: dict[str, Any],
    fuzzy_checker: FuzzyChecker,
    check_noised: pd.DataFrame,
    check_original: pd.DataFrame,
    shared_idx: pd.Index[int],
) -> None:
    config_tree = get_configuration(config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)

        # Check that originally missing data remained missing
        originally_missing_idx = check_original.index[check_original[col.name].isna()]
        assert check_noised.loc[originally_missing_idx, col.name].isna().all()

        # Check for noising where applicable
        to_compare_idx = shared_idx.difference(originally_missing_idx)
        if col.noise_types:
            different_check: npt.NDArray[np.bool_] = np.array(
                check_original.loc[to_compare_idx, col.name].values
                != check_noised.loc[to_compare_idx, col.name].values
            )
            #with check:
            #    assert different_check.any()
            check.is_true(different_check.any())

            noise_level = different_check.sum()

            # Validate column noise level
            validate_column_noise_level(
                dataset_name=dataset_name,
                check_data=check_original,
                check_idx=to_compare_idx,
                noise_level=noise_level,
                col=col,
                config=config_tree,
                fuzzy_name="test_column_noising",
                validator=fuzzy_checker,
            )
        else:  # No noising - should be identical
            same_check: npt.NDArray[np.bool_] = np.array(
                check_original.loc[to_compare_idx, col.name].values
                == check_noised.loc[to_compare_idx, col.name].values
            )

            assert same_check.all()


def run_omit_row_or_do_not_respond_tests(
    dataset_name: str,
    config: dict[str, Any],
    original_data: pd.DataFrame,
    noised_data: pd.DataFrame,
) -> None:
    noise_config: NoiseConfiguration = get_configuration(config)
    noise_types = [
        noise_type
        for noise_type in [NOISE_TYPES.omit_row.name, NOISE_TYPES.do_not_respond.name]
        if noise_config.has_noise_type(dataset_name, noise_type)
    ]

    if dataset_name in [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
    ]:
        # Census and household surveys have do_not_respond and omit_row.
        # For all other datasets they are mutually exclusive
        assert len(noise_types) == 2
    else:
        assert len(noise_types) < 2
    if not noise_types:  # Check that there are no missing indexes
        assert noised_data.index.symmetric_difference(original_data.index).empty
    else:  # Check that there are some omissions
        # TODO: assert levels are as expected
        assert noised_data.index.difference(original_data.index).empty
        assert not original_data.index.difference(noised_data.index).empty

def validate_column_noise_level(
    dataset_name: str,
    check_data: pd.DataFrame,
    check_idx: pd.Index[int],
    noise_level: int,
    col: Column,
    config: NoiseConfiguration,
    fuzzy_name: str,
    validator: FuzzyChecker,
) -> None:
    """
    This helper function iterates through all column noise types for a particular column
    and calculates the expected noise level for each. It then accumulates the expected
    noise level as we layer more noise types on top of each other.
    """
    includes_token_noising = [
        noise_type.name
        for noise_type in COLUMN_NOISE_TYPES
        if config.has_parameter(
            dataset_name, noise_type.name, Keys.TOKEN_PROBABILITY, col.name
        )
        or config.has_parameter(
            dataset_name, noise_type.name, Keys.ZIPCODE_DIGIT_PROBABILITIES, col.name
        )
    ]

    # Calculate expected noise (target proportion for fuzzy checker)
    not_noised = 1.0
    for col_noise_type in col.noise_types:
        if col_noise_type.name not in includes_token_noising:
            not_noised = not_noised * (1 - CELL_PROBABILITY)
        else:
            if col_noise_type.name == NOISE_TYPES.write_wrong_zipcode_digits.name:
                token_probability: list[
                    float
                ] | int | float = config.get_zipcode_digit_probabilities(
                    dataset_name, col.name
                )
            else:
                token_probability = config.get_token_probability(
                    dataset_name, col_noise_type.name, col.name
                )

            # Get number of tokens per string to calculate expected proportion
            tokens_per_string_getter: Callable[
                ..., pd.Series[int] | int
            ] = TOKENS_PER_STRING_MAPPER.get(
                col_noise_type.name, lambda x: x.astype(str).str.len()
            )
            tokens_per_string: pd.Series[int] | int = tokens_per_string_getter(
                check_data.loc[check_idx, col.name]
            )

            # Calculate probability no token is noised
            if isinstance(token_probability, list):
                # Calculate write wrong zipcode average digits probability any token is noise
                avg_probability_any_token_noised = 1 - math.prod(
                    [1 - p for p in token_probability]
                )
            else:
                assert isinstance(tokens_per_string, pd.Series)
                avg_probability_any_token_noised = (
                    1 - (1 - token_probability) ** tokens_per_string
                ).mean()

            # This is accumulating not_noised over all noise types
            not_noised = not_noised * (
                1 - avg_probability_any_token_noised * CELL_PROBABILITY
            )

    expected_noise = 1 - not_noised

    try:
        # Fuzzy checker
        validator.fuzzy_assert_proportion(
            name=fuzzy_name,
            observed_numerator=noise_level,
            observed_denominator=len(check_data.loc[check_idx, col.name]),
            target_proportion=expected_noise,
            name_additional=f"{dataset_name}_{col.name}_{col_noise_type.name}",
        )
    except:
        print(f"{dataset_name} and {col.name} have expected {expected_noise} and actual {noise_level / len(check_data.loc[check_idx, col.name])}")


def initialize_dataset_with_sample(dataset_name: str) -> Dataset:
    SEED = 0
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    data_path = paths.SAMPLE_DATA_ROOT / dataset_name / f"{dataset_name}.parquet"
    dataset = Dataset(dataset_schema, pd.read_parquet(data_path), SEED)

    return dataset

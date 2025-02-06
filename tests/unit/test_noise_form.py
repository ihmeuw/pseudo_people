from __future__ import annotations

import random
from collections.abc import Callable
from string import ascii_lowercase
from typing import NamedTuple

import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from pseudopeople.configuration import Keys
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.dataset import Dataset
from pseudopeople.entity_types import ColumnNoiseType
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS, DatasetSchema
from tests.conftest import FuzzyChecker


@pytest.fixture(scope="module")
def dummy_data() -> pd.DataFrame:
    """Create a two-column dummy dataset"""
    random.seed(0)
    num_rows = 1_000_000
    return pd.DataFrame(
        {
            "event_type": [str(x) for x in range(num_rows)],
            "words": [
                "".join(random.choice(ascii_lowercase) for _ in range(4))
                for _ in range(num_rows)
            ],
        }
    )


def get_dummy_config_noise_numbers(dataset_schema: DatasetSchema) -> NoiseConfiguration:
    """Create a dummy configuration that applies all noise functions to a single
    column in the dummy_data fixture. All noise function specs are defined in
    reverse order here compared to how they are to be applied.

    NOTE: this is not a realistic scenario but allows for certain
    types of stress testing.
    """
    config = LayeredConfigTree(
        {
            dataset_schema.name: {
                Keys.COLUMN_NOISE: {
                    "event_type": {
                        NOISE_TYPES.leave_blank.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.choose_wrong_option.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.copy_from_household_member.name: {
                            Keys.CELL_PROBABILITY: 0.01
                        },
                        NOISE_TYPES.swap_month_and_day.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.write_wrong_zipcode_digits.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.04, 0.04, 0.2, 0.36, 0.36],
                        },
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.POSSIBLE_AGE_DIFFERENCES: [1, -1],
                        },
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.use_nickname.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.use_fake_name.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.make_phonetic_errors.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.make_ocr_errors.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.make_typos.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                    },
                },
                Keys.ROW_NOISE: {
                    NOISE_TYPES.do_not_respond.name: {
                        Keys.ROW_PROBABILITY: 0.01,
                    },
                    NOISE_TYPES.omit_row.name: {
                        Keys.ROW_PROBABILITY: 0.01,
                    },
                    NOISE_TYPES.duplicate_with_guardian.name: {
                        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.05,
                        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0.05,
                    },
                },
            },
        }
    )
    return NoiseConfiguration(config)


@pytest.mark.parametrize(
    "dataset_schema",
    list(DATASET_SCHEMAS),
)
def test_noise_order(
    mocker: MockerFixture, dummy_data: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """From docs: "Noising should be applied in the following order: omit_row,
    do_not_respond, duplicate_row, leave_blank, choose_wrong_option,
    copy_from_household_member, swap_month_and_day, write_wrong_zipcode_digits,
    misreport_age, write_wrong_digits, use_nickname, use_fake_name,
    make_phonetic_errors, make_ocr_errors, make_typos
    """
    mock = mocker.MagicMock()
    # Mock the noise_functions functions so that they are not actually called and
    # return the original one-column dataframe (so that it doesn't become a mock
    # object itself after the first mocked function is applied.)
    mocker.patch(
        "pseudopeople.entity_types.get_index_to_noise", return_value=dummy_data.index
    )
    row_noises = [
        NOISE_TYPES.do_not_respond.name,
        NOISE_TYPES.omit_row.name,
        "duplicate_row",
        NOISE_TYPES.duplicate_with_guardian.name,
    ]
    for field in NOISE_TYPES._fields:
        mock.attach_mock(
            mocker.patch(f"pseudopeople.noise_entities.NOISE_TYPES.{field}.noise_function"),
            field,
        )
        if field not in row_noises:
            mock.attach_mock(
                mocker.patch(
                    f"pseudopeople.noise_entities.NOISE_TYPES.{field}.additional_column_getter",
                    return_value=[],
                ),
                field,
            )
            mock.attach_mock(
                mocker.patch(
                    f"pseudopeople.noise_entities.NOISE_TYPES.{field}.noise_level_scaling_function",
                    return_value=1,
                ),
                field,
            )
        if field == NOISE_TYPES.do_not_respond.name:
            mock.attach_mock(
                mocker.patch(
                    f"pseudopeople.noise_entities.NOISE_TYPES.{field}.get_noise_level",
                    return_value=1,
                ),
                field,
            )

    # Get config for dataset
    dummy_config = get_dummy_config_noise_numbers(dataset_schema)
    # Create a Dataset object from the dummy data
    dataset = Dataset(dataset_schema, dummy_data, 0)
    dataset._noise_dataset(dummy_config, NOISE_TYPES)

    # There are multiple calls that are being mocked. This is not precisely identifying the call
    # to the noise function. Instead, it is identifying mocked calls that have the same argument
    # types that the noise function should be receiving.
    call_order = []
    for mocked_call in mock.mock_calls:
        mocked_call_arguments = mocked_call[1]
        if len(mocked_call_arguments) < 3:
            continue
        first_arg_correct_type = isinstance(mocked_call_arguments[0], Dataset)
        second_arg_correct_type = isinstance(mocked_call_arguments[1], NoiseConfiguration)
        third_arg_correct_type = isinstance(mocked_call_arguments[2], pd.Index)
        if first_arg_correct_type and second_arg_correct_type and third_arg_correct_type:
            call_order.append(mocked_call[0])

    row_order = [
        noise_type
        for noise_type in NOISE_TYPES._fields
        if noise_type
        in [
            NOISE_TYPES.duplicate_with_guardian.name,
            NOISE_TYPES.do_not_respond.name,
            NOISE_TYPES.omit_row.name,
            "duplicate_row",
        ]
    ]
    column_order = [
        NOISE_TYPES.leave_blank.name,
        NOISE_TYPES.choose_wrong_option.name,
        NOISE_TYPES.copy_from_household_member.name,
        NOISE_TYPES.swap_month_and_day.name,
        NOISE_TYPES.write_wrong_zipcode_digits.name,
        NOISE_TYPES.misreport_age.name,
        NOISE_TYPES.write_wrong_digits.name,
        NOISE_TYPES.use_nickname.name,
        NOISE_TYPES.use_fake_name.name,
        NOISE_TYPES.make_phonetic_errors.name,
        NOISE_TYPES.make_ocr_errors.name,
        NOISE_TYPES.make_typos.name,
    ]
    expected_call_order = row_order + column_order

    assert expected_call_order == call_order


# TODO: beef this function up
def test_columns_noised(dummy_data: pd.DataFrame) -> None:
    """Test that the noise functions are only applied to the numbers column
    (as specified in the dummy config)
    """
    config_tree = LayeredConfigTree(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "event_type": {
                        NOISE_TYPES.leave_blank.name: {Keys.CELL_PROBABILITY: 0.1},
                    },
                },
            },
        },
    )
    config = NoiseConfiguration(config_tree)
    dataset = Dataset(DATASET_SCHEMAS.census, dummy_data, 0)
    data = dataset.data.copy()
    dataset._noise_dataset(config, [NOISE_TYPES.leave_blank])
    noised_data = dataset.data

    assert (data["event_type"] != noised_data["event_type"]).any()
    assert (data["words"] == noised_data["words"]).all()


@pytest.mark.parametrize(
    "func, dataset_schema",
    [
        (generate_decennial_census, DATASET_SCHEMAS.census),
        (generate_american_community_survey, DATASET_SCHEMAS.acs),
        (generate_current_population_survey, DATASET_SCHEMAS.cps),
        (generate_women_infants_and_children, DATASET_SCHEMAS.wic),
        (generate_social_security, DATASET_SCHEMAS.ssa),
        (generate_taxes_w2_and_1099, DATASET_SCHEMAS.tax_w2_1099),
        (generate_taxes_1040, DATASET_SCHEMAS.tax_1040),
    ],
)
def test_correct_datasets_are_used(
    func: Callable[..., pd.DataFrame], dataset_schema: DatasetSchema, mocker: MockerFixture
) -> None:
    """Test that each interface noise function uses the correct dataset"""
    mock = mocker.patch("pseudopeople.interface._generate_dataset")
    _ = func()

    assert mock.call_args[0][0] == dataset_schema


def test_two_noise_functions_are_independent(
    fuzzy_checker: FuzzyChecker, monkeypatch: MonkeyPatch
) -> None:
    # Make simple config tree to test 2 noise functions work together
    monkeypatch.setattr(
        "pseudopeople.configuration.noise_configuration.COLUMN_NOISE_TYPES", ["alpha", "beta"]
    )
    config_tree = LayeredConfigTree(
        {
            DATASET_SCHEMAS.census.name: {
                "column_noise": {
                    "fake_column_one": {
                        "alpha": {Keys.CELL_PROBABILITY: 0.20},
                        "beta": {Keys.CELL_PROBABILITY: 0.30},
                    },
                    "fake_column_two": {
                        "alpha": {Keys.CELL_PROBABILITY: 0.40},
                        "beta": {Keys.CELL_PROBABILITY: 0.50},
                    },
                },
            }
        }
    )
    config = NoiseConfiguration(config_tree)

    # Mock objects for testing
    def alpha_noise_function(
        dataset_: Dataset,
        _config: NoiseConfiguration,
        to_noise_idx: pd.Index[int],
        column_name: str,
    ) -> None:
        dataset_.data.loc[to_noise_idx, column_name] += "abc"

    def beta_noise_function(
        dataset_: Dataset,
        _config: NoiseConfiguration,
        to_noise_idx: pd.Index[int],
        column_name: str,
    ) -> None:
        dataset_.data.loc[to_noise_idx, column_name] += "123"

    class MockNoiseTypes(NamedTuple):
        ALPHA: ColumnNoiseType = ColumnNoiseType("alpha", alpha_noise_function)
        BETA: ColumnNoiseType = ColumnNoiseType(
            "beta",
            beta_noise_function,
        )

    mock_noise_types = MockNoiseTypes()

    dummy_dataset = pd.DataFrame(
        {
            "fake_column_one": ["cat", "dog", "bird", "bunny", "duck"] * 20_000,
            "fake_column_two": ["shoe", "pants", "shirt", "hat", "sunglasses"] * 20_000,
        }
    )
    dataset = Dataset(DATASET_SCHEMAS.census, dummy_dataset, 0)
    dataset._noise_dataset(
        configuration=config,
        noise_types=mock_noise_types,
    )
    noised_data = dataset.data

    # Get config values for testing
    col1_expected_abc_proportion = (
        config_tree.decennial_census.column_noise.fake_column_one.alpha[Keys.CELL_PROBABILITY]
    )
    col2_expected_abc_proportion = (
        config_tree.decennial_census.column_noise.fake_column_two.alpha[Keys.CELL_PROBABILITY]
    )
    col1_expected_123_proportion = (
        config_tree.decennial_census.column_noise.fake_column_one.beta[Keys.CELL_PROBABILITY]
    )
    col2_expected_123_proportion = (
        config_tree.decennial_census.column_noise.fake_column_two.beta[Keys.CELL_PROBABILITY]
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_one_abc_proportion",
        observed_numerator=noised_data["fake_column_one"].str.contains("abc").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col1_expected_abc_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_two_abc_proportion",
        observed_numerator=noised_data["fake_column_two"].str.contains("abc").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col2_expected_abc_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_one_123_proportion",
        observed_numerator=noised_data["fake_column_one"].str.contains("123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col1_expected_123_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_two_123_proportion",
        observed_numerator=noised_data["fake_column_two"].str.contains("123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col2_expected_123_proportion,
    )

    # Assert columns experience both noise
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_one_abc123_proportion",
        observed_numerator=noised_data["fake_column_one"].str.contains("abc123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col1_expected_abc_proportion * col1_expected_123_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_two_abc123_proportion",
        observed_numerator=noised_data["fake_column_two"].str.contains("abc123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col2_expected_abc_proportion * col2_expected_123_proportion,
    )
    # Assert expected order of noise application
    assert noised_data["fake_column_one"].str.contains("123abc").sum() == 0
    assert noised_data["fake_column_two"].str.contains("123abc").sum() == 0

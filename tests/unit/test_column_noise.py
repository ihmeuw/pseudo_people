# mypy: disable-error-code="unused-ignore"
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.dataset import Dataset
from pseudopeople.entity_types import ColumnNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS
from pseudopeople.utilities import (
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
    to_string,
)

CHARACTERS_LIST = [
    "A",
    "test123",
    "oF0cr",
    "erR0r5",
    "For456",
    "QUality",
    "contro1",
    "In789",
    "Pseud0peop12E",
]
PHONETIC_STRESS_TEST_LIST = [
    "~",  # No tokens -- note that the tilde character is never noised (from or to) by OCR or phonetic
    "oo",  # Single token
    "~~oo~~",  # Single token, with buffer
    "oooo",  # Only overlapping tokens
    "~~~oooo~~ooo~~~",  # Only overlapping tokens, with buffer
    "ach",  # Overlapping tokens whose corruptions overlap with themselves
    "~~~ach~ach~~~",  # Overlapping tokens whose corruptions overlap with themselves, with buffer
]
PHONETIC_STRESS_TEST_PATHWAYS = {
    # Tuples of (token noised, token not noised)
    "~": {"~": (0, 0)},
    "oo": {
        "oo": (0, 1),
        "u": (1, 0),
    },
    "~~oo~~": {
        "~~oo~~": (0, 1),
        "~~u~~": (1, 0),
    },
    "oooo": {
        "oooo": (0, 3),  # Nothing noised
        "uoo": (1, 1),  # Only first noised
        "ouo": (1, 1),  # Only second noised
        "oou": (1, 2),  # Only third noised
        "uu": (2, 0),  # First and third noised
    },
    "~~~oooo~~ooo~~~": {
        # In each of these additions, the first number represents the number
        # in the first set of o's and the second number represents the number
        # in the second set.
        "~~~oooo~~ooo~~~": (0 + 0, 3 + 2),  # Nothing noised; nothing noised
        "~~~uoo~~ooo~~~": (1 + 0, 1 + 2),  # Only first noised; nothing noised
        "~~~ouo~~ooo~~~": (1 + 0, 1 + 2),  # Only second noised; nothing noised
        "~~~oou~~ooo~~~": (1 + 0, 2 + 2),  # Only third noised; nothing noised
        "~~~uu~~ooo~~~": (2 + 0, 0 + 2),  # First and third noised; nothing noised
        "~~~oooo~~uo~~~": (0 + 1, 3 + 0),  # Nothing noised; only first noised
        "~~~uoo~~uo~~~": (1 + 1, 1 + 0),  # Only first noised; only first noised
        "~~~ouo~~uo~~~": (1 + 1, 1 + 0),  # Only second noised; only first noised
        "~~~oou~~uo~~~": (1 + 1, 2 + 0),  # Only third noised; only first noised
        "~~~uu~~uo~~~": (2 + 1, 0 + 0),  # First and third noised; only first noised
        "~~~oooo~~ou~~~": (0 + 1, 3 + 1),  # Nothing noised; only second noised
        "~~~uoo~~ou~~~": (1 + 1, 1 + 1),  # Only first noised; only second noised
        "~~~ouo~~ou~~~": (1 + 1, 1 + 1),  # Only second noised; only second noised
        "~~~oou~~ou~~~": (1 + 1, 2 + 1),  # Only third noised; only second noised
        "~~~uu~~ou~~~": (2 + 1, 0 + 1),  # First and third noised; only second noised
    },
    # Only enumerate pathways for non-buffer version
    "ach": {
        "ach": (0, 4),  # ach, ch, c, h
        "k": (1, 0),
        "ax": (1, 1),
        "akh": (1, 3),
        "ac": (1, 3),
        "ak": (2, 2),
    },
}
OCR_STRESS_TEST_LIST = [
    "~",  # No tokens
    "YDUu",  # Single-character tokens
    "~~~YD~Uu~~~",  # Single-character tokens, with buffer
    "LII-IIJ",  # Non-overlapping multi-character tokens
    "~~~LI~~I-I~IJ~~~",  # Non-overlapping multi-character tokens, with buffer
    "LI-I",  # Overlapping tokens
    "~~~LI-I~LI-I~~~",  # Overlapping tokens, with buffer
]
OCR_STRESS_TEST_PATHWAYS = {
    # Tuples of (token noised, token not noised)
    "~": {"~": (0, 0)},
    "YDUu": {
        "YDUu": (0, 4),
        "VDUu": (1, 3),
        "YOUu": (1, 3),
        "YDVu": (1, 3),
        "YDUv": (1, 3),
        "VOUu": (2, 2),
        "VDVu": (2, 2),
        "VDUv": (2, 2),
        "YOVu": (2, 2),
        "YOUv": (2, 2),
        "YDVv": (2, 2),
        "YOVv": (3, 1),
        "VDVv": (3, 1),
        "VOUv": (3, 1),
        "VOVu": (3, 1),
        "VOVv": (4, 0),
    },
    "~~~YD~Uu~~~": {
        "~~~YD~Uu~~~": (0, 4),
        "~~~VD~Uu~~~": (1, 3),
        "~~~YO~Uu~~~": (1, 3),
        "~~~YD~Vu~~~": (1, 3),
        "~~~YD~Uv~~~": (1, 3),
        "~~~VO~Uu~~~": (2, 2),
        "~~~VD~Vu~~~": (2, 2),
        "~~~VD~Uv~~~": (2, 2),
        "~~~YO~Vu~~~": (2, 2),
        "~~~YO~Uv~~~": (2, 2),
        "~~~YD~Vv~~~": (2, 2),
        "~~~YO~Vv~~~": (3, 1),
        "~~~VD~Vv~~~": (3, 1),
        "~~~VO~Uv~~~": (3, 1),
        "~~~VO~Vu~~~": (3, 1),
        "~~~VO~Vv~~~": (4, 0),
    },
    "LII-IIJ": {
        "LII-IIJ": (0, 3),  # LI, I-I, IJ
        "UI-IIJ": (1, 2),
        "LIHIJ": (1, 2),
        "LII-IU": (1, 2),
        "UHIJ": (2, 1),
        "UI-IU": (2, 1),
        "UHIJ": (2, 1),
        "LIHU": (2, 1),
        "UI-IU": (2, 1),
        "LIHU": (2, 1),
        "UHU": (3, 0),
    },
    "LI-I": {
        "LI-I": (0, 2),  # LI, I-I
        "U-I": (1, 0),
        "LH": (1, 1),
    },
}
INTEGERS_LIST = [12345, 67890, 54321, 918273, 987654]
FIRST_NAMES = ["Abigail", "Catherine", "Bill", "Fake name"]
LAST_NAMES = ["Johnson", "Smith", "Gates", "Lastname"]
STRING_LIST = [
    "fo1",
    "fo2",
    "fo3",
    "Unit 1A",
    "1234",
    "12/31/2020",
    "a1b2c3",
    "100000.00",
    "123-45-6789",
]


@pytest.fixture(scope="module")
def dummy_dataset() -> pd.DataFrame:
    num_simulants = 1_000_000
    dummy_idx = pd.Index(range(num_simulants))

    # Add a column of integer strings
    integer_series = pd.Series(
        [str(x) for x in INTEGERS_LIST] * int(num_simulants / len(INTEGERS_LIST))
    )
    # integer_series = pd.Series(["Jenny 867-5309", "foo"]*int(num_simulants/2))
    # Add missing data from `leave_blanks` function
    missing_idx = pd.Index([x for x in dummy_idx if x % 3 == 0])
    integer_series.loc[missing_idx] = ""

    # Add a column of character strings
    character_list = CHARACTERS_LIST + [""]
    character_series = pd.Series(character_list * int(num_simulants / len(character_list)))

    # Add a categorical series state column
    states_list = ["CA", "WA", "FL", "OR", "CO", "TX", "NY", "VA", "AZ", "''"]
    states = pd.Series(states_list * int(num_simulants / len(states_list)))

    # Add age col by converting integer_series
    maximum_age = 120
    ages = integer_series.apply(pd.to_numeric, args=("coerce",))
    ages = ages / ages.max() * (maximum_age + 1)
    ages = ages.astype(float).round()

    # Add a string_series column of mixed letters and numbers
    string_series = pd.Series(
        (STRING_LIST + [""]) * int(num_simulants / (len(STRING_LIST) + 1))
    )
    zipcodes = ["12345", "98765", "02468", "13579", ""]
    zipcode_series = pd.Series(zipcodes * int(num_simulants / len(zipcodes)))
    first_names = FIRST_NAMES + [""]
    first_name_series = pd.Series(first_names * int(num_simulants / len(first_names)))
    last_names = LAST_NAMES + [""]
    last_name_series = pd.Series(last_names * int(num_simulants / len(last_names)))
    event_date_list = ["01/25/1990", "05/30/1995", "10/01/2000", "12/31/2010", np.nan]
    event_date_series = pd.Series(event_date_list * int(num_simulants / len(event_date_list)))
    date_of_birth_list = ["01/31/1950", "05/01/1990", "10/01/2000", "12/31/2010", np.nan]
    date_of_birth_series = pd.Series(
        date_of_birth_list * int(num_simulants / len(date_of_birth_list))
    )
    copy_age_list = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, np.nan, 100.0]
    copy_age_series = pd.Series(copy_age_list * int(num_simulants / len(copy_age_list)))

    phonetic_stress_test_series = pd.Series(
        (PHONETIC_STRESS_TEST_LIST + [""])
        * int(num_simulants / (len(PHONETIC_STRESS_TEST_LIST) + 1))
    )
    ocr_stress_test_series = pd.Series(
        (OCR_STRESS_TEST_LIST + [""]) * int(num_simulants / (len(OCR_STRESS_TEST_LIST) + 1))
    )

    return pd.DataFrame(
        {
            "numbers": integer_series,
            "characters": character_series,
            "state": states,
            "age": ages,
            "street_number": string_series,
            "zipcode": zipcode_series,
            "first_name": first_name_series,
            "last_name": last_name_series,
            "event_date": event_date_series,
            "date_of_birth": date_of_birth_series,
            "copy_age": copy_age_series,
            "phonetic_stress_test": phonetic_stress_test_series,
            "ocr_stress_test": ocr_stress_test_series,
        }
    )


@pytest.fixture(scope="module")
def categorical_series() -> pd.Series[str]:
    return pd.Series(
        ["CA", "WA", "FL", "OR", "CO", "TX", "NY", "VA", "AZ", "''"] * 100_000, name="state"
    )


@pytest.fixture(scope="module")
def string_series() -> pd.Series[str]:
    return pd.Series(
        ["Unit 1A", "1234", "12/31/2020", "a1b2c3", "100000.00", "123-45-6789", ""] * 100_000,
        name="random_strings",
    )


@pytest.fixture()
def dataset(dummy_dataset: pd.DataFrame) -> Dataset:
    # We can't have this be a session scope because then we will be operating on the same object
    # across unit tests. We need to be able to modify the original data in each test.
    df = dummy_dataset.copy()
    census = DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name)
    dataset = Dataset(census, df, 0)

    return dataset


@pytest.fixture()
def dataset_same_seed(dummy_dataset: pd.DataFrame) -> Dataset:
    df = dummy_dataset.copy()
    census = DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name)
    dataset = Dataset(census, df, 0)

    return dataset


@pytest.fixture()
def dataset_different_seed(dummy_dataset: pd.DataFrame) -> Dataset:
    # We can't have this be a session scope because then we will be operating on the same object
    # across unit tests. We need to be able to modify the original data in each test.
    df = dummy_dataset.copy()
    census = DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name)
    dataset = Dataset(census, df, 1)

    return dataset


def test_leave_blank(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    column_name = "zipcode"
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column_name: {
                        NOISE_TYPES.leave_blank.name: {
                            Keys.CELL_PROBABILITY: 0.25,
                        },
                    },
                },
            },
        }
    )

    data = dataset.data[[column_name]]
    NOISE_TYPES.leave_blank(dataset, config, column_name)
    noised_data: pd.Series[str] = dataset.data[column_name]
    # Calculate newly missing data, ie data that didn't come in as already missing
    data = data.squeeze()
    is_not_missing = (data.notna()) & (data != "")
    orig_non_missing_idx = is_not_missing[is_not_missing].index
    newly_missing_idx = noised_data.index[
        (noised_data.index.isin(orig_non_missing_idx)) & (noised_data.isna())
    ]

    # Check for expected noise level
    expected_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.leave_blank.name, column_name
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="leave_blank",
        observed_numerator=len(newly_missing_idx),
        observed_denominator=len(orig_non_missing_idx),
        target_proportion=expected_noise,
    )

    # Check that un-noised values are unchanged
    not_noised_idx = noised_data.index[noised_data.notna()]
    assert (data[not_noised_idx] == noised_data[not_noised_idx]).all()


def test_choose_wrong_option(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    config: NoiseConfiguration = get_configuration()
    data = dataset.data["state"].copy()
    NOISE_TYPES.choose_wrong_option(dataset, config, "state")
    noised_data = dataset.data["state"]
    # Check for expected noise level
    expected_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.choose_wrong_option.name, "state"
    )
    # todo: Update when choose_wrong_options uses exclusive resampling
    # Get real expected noise to account for possibility of noising with original value
    # Here we have a a possibility of choosing any of the 50 states for our categorical series fixture
    actual_noise = (noised_data != data).sum()
    original_empty_idx = data.index[data == ""]
    noised_empty_idx = noised_data.index[noised_data == ""]
    fuzzy_checker.fuzzy_assert_proportion(
        name="choose_wrong_option",
        observed_numerator=actual_noise,
        observed_denominator=(len(data) - len(original_empty_idx)),
        target_proportion=expected_noise,
    )
    pd.testing.assert_index_equal(original_empty_idx, noised_empty_idx)


def test_generate_copy_from_household_member(
    dataset: Dataset, fuzzy_checker: FuzzyChecker
) -> None:
    config: NoiseConfiguration = get_configuration()
    original_data = dataset.data[["age", "copy_age"]]
    NOISE_TYPES.copy_from_household_member(dataset, config, "age")
    noised_data = dataset.data["age"]
    # Check for expected noise level
    expected_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.copy_from_household_member.name, "age"
    )
    original_missing_idx = original_data.index[original_data["age"].isnull()]
    eligible_for_noise_idx = original_data.index.difference(original_missing_idx)
    data = original_data["age"]
    actual_noise = (noised_data[eligible_for_noise_idx] != data[eligible_for_noise_idx]).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_copy_from_household_member",
        observed_numerator=actual_noise,
        observed_denominator=len(eligible_for_noise_idx),
        target_proportion=expected_noise,
    )

    # Noised values should be the same as the copy column
    was_noised_series = noised_data[eligible_for_noise_idx] != data[eligible_for_noise_idx]
    noised_idx = was_noised_series[was_noised_series].index
    assert (
        original_data.loc[noised_idx, COPY_HOUSEHOLD_MEMBER_COLS["age"]]
        == noised_data.loc[noised_idx]
    ).all()
    not_noised_idx = eligible_for_noise_idx.difference(noised_idx)
    assert (original_data.loc[not_noised_idx, "age"] == noised_data.loc[not_noised_idx]).all()
    assert noised_data.loc[original_missing_idx].isnull().all()


def test_swap_months_and_days(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    for col in ["event_date", "date_of_birth"]:
        data = dataset.data[col].copy()
        config: NoiseConfiguration = get_configuration()

        cell_probability = {"event_date": 0.25, "date_of_birth": 0.01}[col]

        config._update(
            {
                DATASET_SCHEMAS.census.name: {
                    Keys.COLUMN_NOISE: {
                        col: {
                            NOISE_TYPES.swap_month_and_day.name: {
                                Keys.CELL_PROBABILITY: cell_probability,
                            },
                        },
                    },
                },
            }
        )
        expected_noise: float = config.get_cell_probability(
            DATASET_SCHEMAS.census.name, NOISE_TYPES.swap_month_and_day.name, col
        )

        NOISE_TYPES.swap_month_and_day(dataset, config, col)
        noised_data = dataset.data[col]
        # Confirm missing data remains missing
        orig_missing = data.isna()
        assert (noised_data[orig_missing].isna()).all()
        assert (data[~orig_missing].str[6:] == noised_data[~orig_missing].str[6:]).all()
        actual_noise = (data[~orig_missing] != noised_data[~orig_missing]).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="swap_months_and_days",
            observed_numerator=actual_noise,
            observed_denominator=len(data[~orig_missing]),
            target_proportion=expected_noise,
        )


def test_write_wrong_zipcode_digits(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    column_name: str = "zipcode"
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "zipcode": {
                        NOISE_TYPES.write_wrong_zipcode_digits.name: {
                            Keys.CELL_PROBABILITY: 0.5,
                            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.3, 0.3, 0.4, 0.5, 0.5],
                        },
                    },
                },
            },
        }
    )

    # Get configuration values for each piece of 5 digit zipcode
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_zipcode_digits.name, column_name
    )
    token_probability: list[float] = config.get_zipcode_digit_probabilities(
        dataset.dataset_schema.name, column_name
    )
    data = dataset.data[column_name].copy()
    NOISE_TYPES.write_wrong_zipcode_digits(dataset, config, column_name)
    noised_data = dataset.data[column_name]
    # Confirm missing data remains missing
    orig_missing = data == ""
    assert (noised_data[orig_missing] == "").all()
    # Check noise for each digits position matches expected noise
    for i in range(5):
        digit_prob: float = config.get_zipcode_digit_probabilities(
            dataset.dataset_schema.name, column_name
        )[i]
        actual_noise = (data[~orig_missing].str[i] != noised_data[~orig_missing].str[i]).sum()
        expected_noise = digit_prob * cell_probability
        fuzzy_checker.fuzzy_assert_proportion(
            name="write_wrong_zipcode_digits",
            observed_numerator=actual_noise,
            observed_denominator=len(data[~orig_missing]),
            target_proportion=expected_noise,
        )

    # Validate whole column at once
    actual_noise = (data[~orig_missing] != noised_data[~orig_missing]).sum()
    avg_probability_any_token_noised = 1 - math.prod([1 - p for p in token_probability])
    probability_not_noised = 1 - avg_probability_any_token_noised * cell_probability
    exepected_noised = 1 - probability_not_noised
    fuzzy_checker.fuzzy_assert_proportion(
        name="write_wrong_zipcode_digits_series",
        observed_numerator=actual_noise,
        observed_denominator=len(data[~orig_missing]),
        target_proportion=exepected_noised,
    )


def test_miswrite_ages_default_config(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    """Test that miswritten ages are appropriately handled, including
    no perturbation probabilities defaults to uniform distribution,
    perturbation probabilities"""
    config: NoiseConfiguration = get_configuration()
    data = dataset.data["age"].copy()
    NOISE_TYPES.misreport_age(dataset, config, "age")
    noised_data = dataset.data["age"]

    # Check for expected noise level
    not_missing_idx = data.index[data.notnull()]
    expected_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.misreport_age.name, "age"
    )
    actual_noise = (noised_data[not_missing_idx] != data[not_missing_idx]).sum()
    # NOTE: the expected noise calculated above does not account for the fact that
    # if a perturbed age ends up being the same as the original age, then 1 is subtracted.
    fuzzy_checker.fuzzy_assert_proportion(
        name="misreport_age",
        observed_numerator=actual_noise,
        observed_denominator=len(data[not_missing_idx]),
        target_proportion=expected_noise,
    )

    # Check that missing data remains missing
    original_missing_idx = data.index[data == ""]
    noised_missing_idx = noised_data.index[noised_data == ""]
    pd.testing.assert_index_equal(original_missing_idx, noised_missing_idx)

    # Check that there are no negative ages generated
    assert noised_data[not_missing_idx].astype(int).min() >= 0


def test_miswrite_ages_uniform_probabilities(fuzzy_checker: FuzzyChecker) -> None:
    """Test that a list of perturbations passed in results in uniform probabilities"""
    num_rows = 100_000
    original_age = 25
    perturbations: list[int] = [-2, -1, 1]

    config: NoiseConfiguration = get_configuration(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        }
    )

    data = pd.Series([str(original_age)] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    dataset = Dataset(DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name), df, 0)
    NOISE_TYPES.misreport_age(dataset, config, "age")
    noised_data = dataset.data["age"]
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.misreport_age.name, "age"
    )
    expected_noise = (1 / len(perturbations)) * cell_probability
    for perturbation in perturbations:
        actual_noise = (noised_data.astype(int) - original_age == perturbation).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="misreport_age_uniform_probabilities",
            observed_numerator=actual_noise,
            observed_denominator=len(noised_data),
            target_proportion=expected_noise,
            name_additional=f"perturbation_{perturbation}",
        )


def test_miswrite_ages_provided_probabilities(
    dataset: Dataset, fuzzy_checker: FuzzyChecker
) -> None:
    """Test that provided age perturbation probabilites are handled"""
    num_rows = 100_000
    original_age = 25
    perturbations: dict[int, float] = {-1: 0.1, 1: 0.9}

    config: NoiseConfiguration = get_configuration(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 0.6,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )

    data = pd.Series([str(original_age)] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    dataset = Dataset(DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name), df, 0)
    NOISE_TYPES.misreport_age(dataset, config, "age")
    noised_data = dataset.data["age"]
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.misreport_age.name, "age"
    )
    for perturbation in perturbations:
        expected_noise = perturbations[perturbation] * cell_probability
        actual_noise = (noised_data.astype(int) - original_age == perturbation).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="misreport_age_provided_probabilities",
            observed_numerator=actual_noise,
            observed_denominator=len(data),
            target_proportion=expected_noise,
        )


def test_miswrite_ages_handles_perturbation_to_same_age() -> None:
    """Tests an edge case. It's possible that after an age is perturbed it ends
    up being the original age. In that case, subtract 1. eg, an age of 1 that is
    perturbed -2 becomes -1. But we cannot have negative so we flip the sign to +1.
    But that's the same as the original age and so should become 1-1=0.
    """
    num_rows = 100
    age = 1.0
    perturbations: list[int] = [-2]  # This will cause -1 which will be flipped to +1

    config: NoiseConfiguration = get_configuration(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 0.3,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )

    data = pd.Series([age] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    dataset = Dataset(DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name), df, 0)
    NOISE_TYPES.misreport_age(dataset, config, "age")
    noised_data = dataset.data["age"]
    noised_mask = noised_data != data
    assert (noised_data[noised_mask] == 0).all()


def test_miswrite_ages_flips_negative_to_positive() -> None:
    """Test that any ages perturbed to <0 are reflected to positive values"""
    num_rows = 100
    age = 3.0
    perturbations: list[int] = [-7]  # This will cause -4 and should flip to +4

    config: NoiseConfiguration = get_configuration(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 0.6,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )

    data = pd.Series([age] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    dataset = Dataset(DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name), df, 0)
    NOISE_TYPES.misreport_age(dataset, config, "age")
    noised_data = dataset.data["age"]
    noised_mask = noised_data != data
    assert (noised_data[noised_mask] == 4).all()


def test_write_wrong_digits_robust(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    """
    Validates that only numeric characters are noised in a series at a provided noise level.
    """
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "street_number": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )

    p_row_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_digits.name, "street_number"
    )
    p_token_noise: float = config.get_token_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_digits.name, "street_number"
    )
    data = dataset.data["street_number"].copy()
    # Note: I changed this column from string_series to street number. It has several string formats
    # containing both numeric and alphabetically string characters.
    NOISE_TYPES.write_wrong_digits(dataset, config, "street_number")
    noised_data = dataset.data["street_number"]

    # Get masks for helper groups, each string in categorical string purpose is to mimic possible string types
    empty_str = data == ""
    ambig_str = data.str.len() == 3
    unit_number = data == "Unit 1A"
    id_number = data == "1234"
    alt_str = data == "a1b2c3"
    income = data == "100000.00"
    date_of_birth = data == "12/31/2020"
    ssn = data == "123-45-6789"
    expected_noise = p_row_noise * p_token_noise

    # Check empty strings havent changed
    assert (noised_data[empty_str] == "").all()
    # Assert string length doesn't change after noising
    assert (data.str.len() == noised_data.str.len()).all()

    for i in range(3):  # "fo1", "fo2", "fo3"
        if i == 2:
            actual_noise = (data[ambig_str].str[i] != noised_data[ambig_str].str[i]).sum()
            fuzzy_checker.fuzzy_assert_proportion(
                name="write_wrong_digits_robust_ambig_str",
                observed_numerator=actual_noise,
                observed_denominator=len(data[ambig_str]),
                target_proportion=expected_noise,
            )
        else:
            assert (data[ambig_str].str[i] == noised_data[ambig_str].str[i]).all()

    for i in range(4):  # "1234"
        actual_noise = (data[id_number].str[i] != noised_data[id_number].str[i]).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="write_wrong_digits_robust_id_number",
            observed_numerator=actual_noise,
            observed_denominator=len(data[id_number]),
            target_proportion=expected_noise,
        )
        assert (noised_data[id_number].str[i].str.isdigit()).all()

    for i in range(6):  # "a1b2c3"
        if i % 2 == 0:
            assert (data[alt_str].str[i] == noised_data[alt_str].str[i]).all()
        else:
            actual_noise = (data[alt_str].str[i] != noised_data[alt_str].str[i]).sum()
            fuzzy_checker.fuzzy_assert_proportion(
                name="write_wrong_digits_robust_alt_str",
                observed_numerator=actual_noise,
                observed_denominator=len(data[alt_str]),
                target_proportion=expected_noise,
            )
            assert (noised_data[alt_str].str[i].str.isdigit()).all()

    for i in range(7):  # "Unit 1A"
        if i == 5:
            actual_noise = (data[unit_number].str[i] != noised_data[unit_number].str[i]).sum()
            fuzzy_checker.fuzzy_assert_proportion(
                name="write_wrong_digits_robust_unit_number",
                observed_numerator=actual_noise,
                observed_denominator=len(data[unit_number]),
                target_proportion=expected_noise,
            )
            assert (noised_data[unit_number].str[i].str.isdigit()).all()
        else:
            assert (data[unit_number].str[i] == noised_data[unit_number].str[i]).all()

    for i in range(9):  # "100000.00"
        if i == 6:
            assert (data[income].str[i] == noised_data[income].str[i]).all()
        else:
            actual_noise = (data[income].str[i] != noised_data[income].str[i]).sum()
            fuzzy_checker.fuzzy_assert_proportion(
                name="write_wrong_digits_robust_income",
                observed_numerator=actual_noise,
                observed_denominator=len(data[income]),
                target_proportion=expected_noise,
            )
            assert (noised_data[income].str[i].str.isdigit()).all()

    for i in range(10):  # "12/31/2020"
        if i in [2, 5]:
            assert (data[date_of_birth].str[i] == noised_data[date_of_birth].str[i]).all()
        else:
            actual_noise = (
                data[date_of_birth].str[i] != noised_data[date_of_birth].str[i]
            ).sum()
            fuzzy_checker.fuzzy_assert_proportion(
                name="write_wrong_digits_robust_date_of_birth",
                observed_numerator=actual_noise,
                observed_denominator=len(data[date_of_birth]),
                target_proportion=expected_noise,
            )
            assert (noised_data[date_of_birth].str[i].str.isdigit()).all()

    for i in range(11):  # "123-45-6789"
        if i in [3, 6]:
            assert (data[ssn].str[i] == noised_data[ssn].str[i]).all()
        else:
            actual_noise = (data[ssn].str[i] != noised_data[ssn].str[i]).sum()
            fuzzy_checker.fuzzy_assert_proportion(
                name="write_wrong_digits_robust_ssn",
                observed_numerator=actual_noise,
                observed_denominator=len(data[ssn]),
                target_proportion=expected_noise,
            )
            assert (noised_data[ssn].str[i].str.isdigit()).all()


def test_write_wrong_digits(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    # This is a quicker (less robust) version of the test above.
    # It only checks that numeric characters are noised at the correct level as
    # a sanity check our noise is of the right magnitude
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "street_number": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )

    expected_cell_noise = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_digits.name, "street_number"
    )
    expected_token_noise: float = config.get_token_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_digits.name, "street_number"
    )
    data = dataset.data["street_number"].copy()
    # Note: I changed this column from string_series to street number. It has several string formats
    # containing both numeric and alphabetically string characters.
    NOISE_TYPES.write_wrong_digits(dataset, config, "street_number")
    noised_data = dataset.data["street_number"]
    # Validate we do not change any missing data
    missing_mask = data == ""
    assert (noised_data[missing_mask] == "").all()

    # Check expected noise level
    check_original = data[~missing_mask]
    string_series = pd.Series(STRING_LIST)
    # Calculate average number of digits per string in the data
    # Replace no numeric values with nothing
    digits_per_string = string_series.str.replace(r"[^\d]", "", regex=True).str.len()
    avg_probability_any_token_noised = (
        1 - (1 - expected_token_noise) ** digits_per_string
    ).mean()
    check_noised = noised_data[~missing_mask]
    actual_noise = (check_original != check_noised).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="write_wrong_digits",
        observed_numerator=actual_noise,
        observed_denominator=len(check_original),
        target_proportion=expected_cell_noise * avg_probability_any_token_noised,
    )


def test_use_nickname(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    config: NoiseConfiguration = get_configuration()
    expected_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.use_nickname.name, "first_name"
    )
    data = dataset.data["first_name"].copy()
    NOISE_TYPES.use_nickname(dataset, config, "first_name")
    noised_data = dataset.data["first_name"]

    # Validate missing stays missing
    orig_missing = data.isna()
    assert (noised_data[orig_missing].isna()).all()
    # Validate noise level
    actual_noise = (noised_data[~orig_missing] != data[~orig_missing]).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="use_nickname",
        observed_numerator=actual_noise,
        observed_denominator=len(data[~orig_missing]),
        target_proportion=expected_noise,
    )

    # Validation for nicknames
    from pseudopeople.noise_scaling import load_nicknames_data

    nicknames = load_nicknames_data()
    names_list = pd.Series(
        nicknames.apply(lambda row: row.dropna().tolist(), axis=1), index=nicknames.index
    )
    for real_name in data.dropna().unique():
        # Validates names that are not nickname eligible do not not get noised
        if real_name not in names_list.index:
            assert (data.loc[data == real_name] == noised_data[data == real_name]).all()
        else:
            real_name_idx = data.index[data == real_name]
            # Verify options chosen are valid nicknames for original names that were noised
            assert set(noised_data.loc[real_name_idx].dropna()).issubset(
                set(names_list.loc[real_name] + [real_name])
            )
            # Validate we choose the nicknames for each name randomly (equally)
            chosen_nicknames = noised_data.loc[
                real_name_idx.difference(noised_data.index[noised_data == real_name])
            ]
            chosen_nickname_counts = pd.Series(chosen_nicknames.value_counts())
            expected_name_proportion = 1 / len(names_list.loc[real_name])
            for nickname in chosen_nickname_counts.index:
                fuzzy_checker.fuzzy_assert_proportion(
                    name="use_nickname_proportion",
                    observed_numerator=chosen_nickname_counts.loc[nickname],
                    observed_denominator=chosen_nickname_counts.sum(),
                    target_proportion=expected_name_proportion,
                    name_additional=f"for nickname: {nickname} of real name: {real_name}",
                )


@pytest.mark.parametrize("column", ["first_name", "last_name"])
def test_use_fake_name(dataset: Dataset, column: str, fuzzy_checker: FuzzyChecker) -> None:
    """
    Function to test that fake names are noised and replace raw values at a configured percentage
    """
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.use_fake_name.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                        },
                    },
                },
            },
        }
    )

    data = dataset.data[column].copy()
    NOISE_TYPES.use_fake_name(dataset, config, column)
    noised_data = dataset.data[column]

    # Check missing are unchanged
    orig_missing = data == ""
    assert (data[orig_missing] == noised_data[orig_missing]).all()
    # todo: equal across fake values
    # Check noised values
    actual_noise = (data[~orig_missing] != noised_data[~orig_missing]).sum()
    expected_noise: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.use_fake_name.name, column
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="use_fake_name_first_name",
        observed_numerator=actual_noise,
        observed_denominator=len(data[~orig_missing]),
        target_proportion=expected_noise,
    )

    # Get raw fake names lists to check noised values
    fake_names = {"first_name": fake_first_names, "last_name": fake_last_names}
    fake_options = fake_names[column]

    assert noised_data.loc[noised_data != data].isin(fake_options).all()


@pytest.mark.parametrize(
    "column",
    [
        "first_name",
        "last_name",
        "numbers",
        "characters",
        "phonetic_stress_test",
    ],
)
def test_generate_phonetic_errors(
    dataset: Dataset, column: str, fuzzy_checker: FuzzyChecker
) -> None:
    data = dataset.data[column].copy()

    config = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.make_phonetic_errors.name: {
                            Keys.CELL_PROBABILITY: 0.1,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    }
                },
            },
        }
    )
    NOISE_TYPES.make_phonetic_errors(dataset, config, column)
    noised_data = dataset.data[column]

    # Validate we do not change any missing data
    missing_mask = data == ""
    assert (noised_data[missing_mask] == "").all()

    # Check expected noise level
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.make_phonetic_errors.name, column
    )
    token_probability: float = config.get_token_probability(
        dataset.dataset_schema.name, NOISE_TYPES.make_phonetic_errors.name, column
    )
    check_original = data[~missing_mask]
    check_noised = noised_data[~missing_mask]
    actual_noise = (check_original != check_noised).sum()

    if column == "first_name":
        equally_likely_values = pd.Series(FIRST_NAMES)
    elif column == "last_name":
        equally_likely_values = pd.Series(LAST_NAMES)
    elif column == "characters":
        equally_likely_values = pd.Series(CHARACTERS_LIST)
    elif column == "numbers":
        equally_likely_values = pd.Series(INTEGERS_LIST).astype(str)
    elif column == "phonetic_stress_test":
        equally_likely_values = pd.Series(PHONETIC_STRESS_TEST_LIST)

    original_phonetic_tokens = pd.Series(load_phonetic_errors().index)
    # Calculate average number of tokens per string in the data
    tokens_per_string = number_of_tokens_per_string(
        original_phonetic_tokens, equally_likely_values
    )
    avg_probability_any_token_noised = (
        1 - (1 - token_probability) ** tokens_per_string
    ).mean()
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_phonetic_errors",
        observed_numerator=actual_noise,
        observed_denominator=len(check_original),
        target_proportion=cell_probability * avg_probability_any_token_noised,
    )


@pytest.mark.parametrize(
    "pair",
    PHONETIC_STRESS_TEST_PATHWAYS.items(),
)
def test_phonetic_error_values(
    pair: tuple[str, dict[str, tuple[int, ...]]], fuzzy_checker: FuzzyChecker
) -> None:
    string, pathways = pair
    column_name = "first_name"

    data = pd.Series([string] * 100_000, name=column_name)
    config = get_configuration()

    cell_probability = 0.9
    token_probability = 0.3

    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column_name: {
                        NOISE_TYPES.make_phonetic_errors.name: {
                            Keys.CELL_PROBABILITY: cell_probability,
                            Keys.TOKEN_PROBABILITY: token_probability,
                        },
                    }
                },
            },
        }
    )

    df = pd.DataFrame({column_name: data})
    dataset = Dataset(DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name), df, 0)
    NOISE_TYPES.make_phonetic_errors(dataset, config, column_name)
    noised_data = dataset.data[column_name]

    assert noised_data.isin(
        pathways.keys()
    ).all(), f"Unexpected results for {string}: {set(noised_data) - set(pathways.keys())}"
    for result, (num_noised, num_not_noised) in pathways.items():
        pathway_probability = (
            token_probability**num_noised * (1 - token_probability) ** num_not_noised
        )
        probability = pathway_probability * cell_probability
        if result == string:
            # If cell is not selected, we end up with this result
            probability += 1 - cell_probability

        fuzzy_checker.fuzzy_assert_proportion(
            name="test_phonetic_error_values",
            observed_numerator=(noised_data == result).sum(),
            observed_denominator=len(data),
            target_proportion=probability,
            name_additional=f"result {result} from string {string}",
        )


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
        "first_name",
        "last_name",
        "ocr_stress_test",
    ],
)
def test_generate_ocr_errors(
    dataset: Dataset, column: str, fuzzy_checker: FuzzyChecker
) -> None:
    config: NoiseConfiguration = get_configuration()

    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.make_ocr_errors.name: {
                            Keys.CELL_PROBABILITY: 0.1,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                    }
                },
            },
        }
    )

    # Get node
    data = dataset.data[column].copy()
    NOISE_TYPES.make_ocr_errors(dataset, config, column)
    noised_data = dataset.data[column]

    # Validate we do not change any missing data
    missing_mask = data == ""
    assert (data[missing_mask] == noised_data[missing_mask]).all()

    # Check expected noise level
    token_probability: float = config.get_token_probability(
        dataset.dataset_schema.name, NOISE_TYPES.make_ocr_errors.name, column
    )
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.make_ocr_errors.name, column
    )
    # We need to calculate the expected noise. We need to get the average number of tokens per string
    # that can be noised since not all tokens can be noised for OCR errors.
    if column == "first_name":
        equally_likely_values = pd.Series(FIRST_NAMES)
    elif column == "last_name":
        equally_likely_values = pd.Series(LAST_NAMES)
    elif column == "characters":
        equally_likely_values = pd.Series(CHARACTERS_LIST)
    elif column == "numbers":
        equally_likely_values = pd.Series(INTEGERS_LIST).astype(str)
    elif column == "ocr_stress_test":
        equally_likely_values = pd.Series(OCR_STRESS_TEST_LIST)

    ocr_tokens = pd.Series(load_ocr_errors().index)
    tokens_per_string = number_of_tokens_per_string(ocr_tokens, equally_likely_values)
    avg_probability_any_token_noised = (
        1 - (1 - token_probability) ** tokens_per_string
    ).mean()
    check_original = data[~missing_mask]
    check_noised = noised_data[~missing_mask]
    expected_proportion = cell_probability * avg_probability_any_token_noised
    actual_noise = (check_original != check_noised).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_ocr_errors",
        observed_numerator=actual_noise,
        observed_denominator=len(check_original),
        target_proportion=expected_proportion,
    )


@pytest.mark.parametrize(
    "pair",
    OCR_STRESS_TEST_PATHWAYS.items(),
)
def test_ocr_replacement_values(
    pair: tuple[str, dict[str, tuple[int, ...]]], fuzzy_checker: FuzzyChecker
) -> None:
    string, pathways = pair
    column_name = "first_name"

    data = pd.Series([string] * 100_000, name="column")
    cell_probability = 0.9
    token_probability = 0.3
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column_name: {
                        NOISE_TYPES.make_ocr_errors.name: {
                            Keys.CELL_PROBABILITY: cell_probability,
                            Keys.TOKEN_PROBABILITY: token_probability,
                        },
                    }
                },
            },
        }
    )

    df = pd.DataFrame({column_name: data})
    dataset = Dataset(DATASET_SCHEMAS.get_dataset_schema(DATASET_SCHEMAS.census.name), df, 0)
    NOISE_TYPES.make_ocr_errors(dataset, config, column_name)
    noised_data = dataset.data[column_name]

    assert noised_data.isin(
        pathways.keys()
    ).all(), f"Unexpected results for {string}: {set(noised_data) - set(pathways.keys())}"
    for result, (num_noised, num_not_noised) in pathways.items():
        pathway_probability = (
            token_probability**num_noised * (1 - token_probability) ** num_not_noised
        )
        probability = pathway_probability * cell_probability
        if result == string:
            # If cell is not selected, we end up with this result
            probability += 1 - cell_probability

        fuzzy_checker.fuzzy_assert_proportion(
            name="test_ocr_replacement_values",
            observed_numerator=(noised_data == result).sum(),
            observed_denominator=len(data),
            target_proportion=probability,
            name_additional=f"result {result} from string {string}",
        )


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
    ],
)
def test_make_typos(dataset: Dataset, column: str, fuzzy_checker: FuzzyChecker) -> None:
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.make_typos.name: {
                            Keys.CELL_PROBABILITY: 0.1,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                    },
                },
            },
        }
    )

    data = dataset.data[column].copy()
    NOISE_TYPES.make_typos(dataset, config, column)
    noised_data = dataset.data[column]

    not_missing_idx = data.index[(data.notna()) & (data != "")]
    check_original = data.loc[not_missing_idx]
    check_noised = noised_data.loc[not_missing_idx]

    # Check for expected noise level
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.make_typos.name, column
    )
    token_probability: float = config.get_token_probability(
        dataset.dataset_schema.name, NOISE_TYPES.make_typos.name, column
    )
    qwerty_tokens = pd.Series(load_qwerty_errors_data().index)
    data_series = (
        pd.Series(INTEGERS_LIST) if column == "numbers" else pd.Series(CHARACTERS_LIST)
    )
    data_series = data_series.astype(str)
    tokens_per_string = number_of_tokens_per_string(qwerty_tokens, data_series)
    avg_probability_any_token_noised = (
        1 - (1 - token_probability) ** tokens_per_string
    ).mean()
    expected_noise = cell_probability * avg_probability_any_token_noised
    actual_noise = (check_noised != check_original).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="make_typos",
        observed_numerator=actual_noise,
        observed_denominator=len(check_original),
        target_proportion=expected_noise,
    )

    # Check for expected string growth due to keeping original noised token
    assert (check_noised.str.len() >= check_original.str.len()).all()
    # TODO: remove this hard-coding
    p_include_original_token = 0.1
    p_token_does_not_increase_string_length = 1 - token_probability * p_include_original_token
    p_strings_do_not_increase_length = (
        p_token_does_not_increase_string_length**tokens_per_string
    )  # pd.Series
    p_strings_increase_length = (1 - p_strings_do_not_increase_length).mean()
    expected_changed_length = cell_probability * p_strings_increase_length
    actual_changed_length = (check_noised.str.len() != check_original.str.len()).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="make_typos_string_length",
        observed_numerator=actual_changed_length,
        observed_denominator=len(check_original),
        target_proportion=expected_changed_length,
    )

    # Check that we did not touch the missing data
    assert (
        data.loc[~data.index.isin(not_missing_idx)]
        == noised_data.loc[~noised_data.index.isin(not_missing_idx)]
    ).all()


@pytest.mark.parametrize(
    "noise_type, data_col, dataset_name, dataset_col",
    [
        (NOISE_TYPES.leave_blank, "numbers", "decennial_census", "zipcode"),
        (NOISE_TYPES.choose_wrong_option, "state", "decennial_census", "state"),
        (NOISE_TYPES.copy_from_household_member, "age", "decennial_census", "age"),
        (NOISE_TYPES.swap_month_and_day, "event_date", "social_security", "event_date"),
        (NOISE_TYPES.write_wrong_zipcode_digits, "zipcode", "decennial_census", "zipcode"),
        (NOISE_TYPES.misreport_age, "age", "decennial_census", "age"),
        (
            NOISE_TYPES.write_wrong_digits,
            "street_number",
            "decennial_census",
            "street_number",
        ),
        (NOISE_TYPES.use_nickname, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.use_fake_name, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.use_fake_name, "last_name", "decennial_census", "last_name"),
        (NOISE_TYPES.make_phonetic_errors, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.make_ocr_errors, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.make_typos, "numbers", "decennial_census", "zipcode"),
        (NOISE_TYPES.make_typos, "characters", "decennial_census", "street_name"),
    ],
)
def test_seeds_behave_as_expected(
    noise_type: ColumnNoiseType,
    data_col: str,
    dataset_name: str,
    dataset_col: str,
    dataset: Dataset,
    dataset_same_seed: Dataset,
    dataset_different_seed: Dataset,
) -> None:
    """Tests that different seeds produce different results and the same seed
    produces the same results
    """
    if data_col == "todo":
        pytest.skip(reason=f"TODO: implement for {noise_type}")
    noise = noise_type.name
    config: NoiseConfiguration = get_configuration()

    COLS_MISSING_FROM_CONFIG = ["numbers", "event_date", "characters"]

    if data_col in COLS_MISSING_FROM_CONFIG:
        cell_probability = config.get_cell_probability(dataset_name, noise, dataset_col)
        # if we have token probability:
        try:
            token_probability = config.get_token_probability(dataset_name, noise, dataset_col)
        except:
            token_probability = 0.0
        config._update(
            {
                dataset.dataset_schema.name: {
                    Keys.COLUMN_NOISE: {
                        data_col: {
                            noise: {
                                Keys.CELL_PROBABILITY: cell_probability,
                                Keys.TOKEN_PROBABILITY: token_probability,
                            },
                        },
                    },
                },
            }
        )

    if noise == NOISE_TYPES.copy_from_household_member.name:
        data: pd.DataFrame | pd.Series[Any] = dataset.data[
            [data_col, COPY_HOUSEHOLD_MEMBER_COLS[data_col]]
        ].copy()
    else:
        data = dataset.data[[data_col]].copy()

    noise_type(dataset, config, data_col)
    noised_data = dataset.data[data_col]
    noise_type(dataset_same_seed, config, data_col)
    noised_data_same_seed = dataset_same_seed.data[data_col]
    noise_type(dataset_different_seed, config, data_col)
    noised_data_different_seed = dataset_different_seed.data[data_col]
    data = data[data_col]

    assert (noised_data != data).any()
    assert (noised_data.isna() == noised_data_same_seed.isna()).all()
    assert (
        noised_data[noised_data.notna()]
        == noised_data_same_seed[noised_data_same_seed.notna()]
    ).all()
    assert (noised_data != noised_data_different_seed).any()

    # Check that we are in fact getting different noised values
    noised = noised_data.loc[noised_data != data].reset_index(drop=True)
    noised_different_seed = noised_data_different_seed.loc[
        noised_data_different_seed != data
    ].reset_index(drop=True)
    shortest = min(len(noised), len(noised_different_seed))
    assert (noised.iloc[:shortest] != noised_different_seed.iloc[:shortest]).any()


def test_age_write_wrong_digits(dataset: Dataset, fuzzy_checker: FuzzyChecker) -> None:
    # Tests write wrong digits is now applied to age column - albrja(10/23/23)
    config = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )

    data = to_string(dataset.data["age"].copy())
    NOISE_TYPES.write_wrong_digits(dataset, config, "age")
    # Calculate expected noise level
    noised_data = dataset.data["age"]
    missing_mask = data.isnull()
    check_original = data[~missing_mask]
    check_noised = noised_data[~missing_mask]
    expected_noise = (check_original != check_noised).sum()
    cell_probability: float = config.get_cell_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_digits.name, "age"
    )
    token_probability: float = config.get_token_probability(
        dataset.dataset_schema.name, NOISE_TYPES.write_wrong_digits.name, "age"
    )
    tokens_per_string = check_original.astype(str).str.len()
    avg_probability_any_token_noised = (
        1 - (1 - token_probability) ** tokens_per_string
    ).mean()
    fuzzy_checker.fuzzy_assert_proportion(
        name="write_wrong_digits_age",
        observed_numerator=expected_noise,
        observed_denominator=len(check_original),
        target_proportion=cell_probability * avg_probability_any_token_noised,
    )


################
### Wrappers ###
################


def number_of_tokens_per_string(s1: Iterable[str], s2: Iterable[str]) -> pd.Series[int]:
    """
    Calculates the number of tokens in each string of a series.
    s1 is a pd.Series of tokens and we want to see how many tokens exist in each
    string of s2.
    """

    number_of_tokens = pd.Series(0, index=[str(i) for i in s2])
    for token in s1:
        orig_token = str(token)
        for s in s2:
            string = str(s)
            number_of_tokens.loc[string] += occurrences(string, orig_token)

    return number_of_tokens


# https://stackoverflow.com/a/2970542/
def occurrences(string: str, sub: str) -> int:
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count

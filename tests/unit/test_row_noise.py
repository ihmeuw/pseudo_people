import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.dataset import Dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_level import _get_census_omission_noise_levels
from pseudopeople.schema_entities import DATASET_SCHEMAS
from tests.conftest import FuzzyChecker


@pytest.fixture()
def dummy_data() -> pd.DataFrame:
    num_simulants = 1_000_000
    dummy_idx = pd.Index(range(num_simulants))

    numbers_list = list(range(10))
    numbers_series = pd.Series(numbers_list * int(num_simulants / len(numbers_list)))
    word_list = ["word_1", "word_2", "word_3", "word_4", "word5"]
    word_series = pd.Series(word_list * int(num_simulants / len(word_list)))

    return pd.DataFrame(
        {
            "numbers": numbers_series,
            "words": word_series,
        },
        index=dummy_idx,
    )


def test_omit_row(dummy_data: pd.DataFrame, fuzzy_checker: FuzzyChecker) -> None:
    config: NoiseConfiguration = get_configuration()
    dataset = Dataset(DATASET_SCHEMAS.tax_w2_1099, dummy_data, 0)
    NOISE_TYPES.omit_row(dataset, config)
    noised_data1 = dataset.data

    expected_noise_1: float = config.get_row_probability(
        DATASET_SCHEMAS.tax_w2_1099.name, "omit_row"
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_omit_row",
        observed_numerator=len(noised_data1),
        observed_denominator=len(dummy_data),
        target_proportion=1 - expected_noise_1,
    )
    assert set(noised_data1.columns) == set(dummy_data.columns)
    assert (noised_data1.dtypes == dummy_data.dtypes).all()


def test_do_not_respond(
    mocker: MockerFixture, dummy_data: pd.DataFrame, fuzzy_checker: FuzzyChecker
) -> None:
    config: NoiseConfiguration = get_configuration()
    mocker.patch(
        "pseudopeople.noise_level._get_census_omission_noise_levels",
        side_effect=(
            lambda *_: config.get_row_probability(
                DATASET_SCHEMAS.census.name, NOISE_TYPES.do_not_respond.name
            )
        ),
    )

    my_dummy_data = dummy_data.copy()
    my_dummy_data["age"] = 27
    my_dummy_data["sex"] = "Female"
    my_dummy_data["race_ethnicity"] = "Vulcan"
    census = Dataset(DATASET_SCHEMAS.census, my_dummy_data, 0)
    acs = Dataset(DATASET_SCHEMAS.acs, my_dummy_data, 0)
    NOISE_TYPES.do_not_respond(census, config)
    NOISE_TYPES.do_not_respond(acs, config)
    noised_census = census.data
    noised_acs = acs.data
    target_proportion: float = config.get_row_probability(
        DATASET_SCHEMAS.census.name, NOISE_TYPES.do_not_respond.name
    )

    # Test that noising affects expected proportion with expected types
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(my_dummy_data) - len(noised_census),
        observed_denominator=len(my_dummy_data),
        target_proportion=target_proportion,
        name_additional=f"noised_data1",
    )
    assert set(noised_census.columns) == set(my_dummy_data.columns)
    assert (noised_census.dtypes == my_dummy_data.dtypes).all()

    # Check ACS data is scaled properly due to oversampling
    row_probability: float = config.get_row_probability(
        DATASET_SCHEMAS.census.name, NOISE_TYPES.do_not_respond.name
    )
    expected_noise = 0.5 + row_probability / 2
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(my_dummy_data) - len(noised_acs),
        observed_denominator=len(my_dummy_data),
        target_proportion=expected_noise,
        name_additional=f"noised_data2",
    )
    assert set(noised_acs.columns) == set(my_dummy_data.columns)
    assert (noised_acs.dtypes == my_dummy_data.dtypes).all()
    assert len(noised_census) != len(noised_acs)
    assert True


@pytest.mark.parametrize(
    "age, race_ethnicity, sex, expected_level",
    [
        (3, "White", "Female", 0.0091),
        (35, "Black", "Male", 0.0611),
        (55, "Asian", "Female", 0.0),
    ],
)
def test__get_census_omission_noise_levels(
    age: int, race_ethnicity: str, sex: str, expected_level: float
) -> None:
    """Test helper function for do_not_respond noising based on demography of age, race/ethnicity, and sex"""
    pop = pd.DataFrame(
        [[age, race_ethnicity, sex] for i in range(10)],
        index=range(10),
        columns=["age", "race_ethnicity", "sex"],
    )
    result = _get_census_omission_noise_levels(pop)
    assert (np.isclose(result, expected_level, rtol=0.0001)).all()


def test_do_not_respond_missing_columns(dummy_data: pd.DataFrame) -> None:
    """Test do_not_respond raises error when missing required columns."""
    config: NoiseConfiguration = get_configuration()
    census = Dataset(DATASET_SCHEMAS.census, dummy_data, 0)
    with pytest.raises(KeyError, match="race_ethnicity"):
        NOISE_TYPES.do_not_respond(census, config)


def test_guardian_duplication() -> None:
    # We are going to make a small dataframe and update the configuration to noise 100% of the
    # available rows. We will then check that the correct rows were copied with the correct
    # information.
    dummy_data = pd.DataFrame(
        {
            "simulant_id": [str(i) for i in list(range(10))],
            "household_id": [0, 0, 0, 1, 2, 3, 4, 4, 5, 6],
            "housing_type": [
                "Household",
                "Household",
                "Household",
                "College",
                "Military",
                "Household",
                "Household",
                "Household",
                "Household",
                "Household",
            ],
            "age": [10, 10, 10, 10, 10, 17, 50, 50, 50, 50],
            "guardian_1": ["8", "7", "5", "9", "7", "6", np.nan, np.nan, np.nan, np.nan],
            "guardian_2": [
                "9",
                np.nan,
                "8",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "relationship_to_reference_person": ["Fake value"] * 10,
            "year": [2019] * 10,
            "street_name": [
                "Street 0",
                "Street 0",
                "Street 1",
                "Street 1",
                "Street 1",
                "Street 2",
                "Street 3",
                "Street 3",
                "Street 4",
                "Street 5",
            ],
            "street_number": [0, 0, 1, 1, 1, 2, 3, 3, 4, 5],
            "unit_number": [0, 0, 1, 1, 1, 2, 3, 3, 4, 5],
            "city": [
                "City 0",
                "City 0",
                "City 1",
                "City 1",
                "City 1",
                "City 2",
                "City 3",
                "City 3",
                "City 4",
                "City 5",
            ],
            "state": [
                "State 0",
                "State 0",
                "State 1",
                "State 1",
                "State 1",
                "State 2",
                "State 3",
                "State 3",
                "State 4",
                "State 5",
            ],
            "zipcode": [
                00000,
                00000,
                11111,
                11111,
                11111,
                22222,
                33333,
                33333,
                44444,
                55555,
            ],
        }
    )
    # Noise 100% of rows
    config: NoiseConfiguration = get_configuration()
    config._update(
        {
            DATASET_SCHEMAS.census.name: {
                Keys.ROW_NOISE: {
                    NOISE_TYPES.duplicate_with_guardian.name: {
                        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 1,
                        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 1,
                    },
                },
            }
        }
    )
    census = Dataset(DATASET_SCHEMAS.census, dummy_data, 0)
    NOISE_TYPES.duplicate_with_guardian(census, config)
    noised = census.data
    # We know the following since every dependent is duplicated:
    #  - Simulant ids 0, 1, 2, 3, and-5 will all be duplicated
    #  - Simulant ids 5-9 are guardians. The only overlap is simulant id 5,
    #    who is both a dependent and a guardian
    #  - Simulant id 0 and 3 have two guardians, 8 and 9.

    # Check that the correct rows were duplicated. Duplicated returns all instances of True after
    # the first instance
    duplicated = noised.loc[noised["simulant_id"].duplicated()]
    guardians = dummy_data.loc[dummy_data["simulant_id"].isin(dummy_data["guardian_1"])]
    assert len(noised) == len(dummy_data) + len(duplicated)
    assert set(duplicated["simulant_id"].tolist()) == set(["0", "1", "2", "3", "5"])
    # Only duplicate a depedent one time
    assert noised["simulant_id"].value_counts().max() == 2

    # Check address information is copied in new rows
    for i in duplicated.index:
        dependent = duplicated.loc[i]
        for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS:
            guardian_1 = dependent["guardian_1"]
            guardian_2 = dependent["guardian_2"]
            if guardian_2 is np.nan:
                guardians_values = [
                    guardians.loc[guardians["simulant_id"] == guardian_1, column].values[0]
                ]
            else:
                guardians_values = [
                    guardians.loc[guardians["simulant_id"] == guardian_1, column].values[0],
                    guardians.loc[guardians["simulant_id"] == guardian_2, column].values[0],
                ]
            assert dependent[column] in guardians_values

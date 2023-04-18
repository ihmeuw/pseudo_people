from pathlib import Path

import pytest
import yaml

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.generator import DEFAULT_NOISE_VALUES
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, FORMS


def test_get_default_configuration(mocker):
    """Tests that the default configuration can be retrieved."""
    mock = mocker.patch("pseudopeople.configuration.generator.ConfigTree")
    _ = get_configuration()
    mock.assert_called_once_with(layers=["baseline", "default", "user"])


def test_default_configuration_structure():
    """Test that the default configuration structure is correct"""
    config = get_configuration()
    # Check forms
    assert set(f.name for f in FORMS) == set(config.keys())
    for form in FORMS:
        # Check row noise
        for row_noise in form.row_noise_types:
            config_probability = config[form.name][Keys.ROW_NOISE][row_noise.name][
                Keys.PROBABILITY
            ]
            default_probability = (
                DEFAULT_NOISE_VALUES.get(form.name, {})
                .get(Keys.ROW_NOISE, {})
                .get(row_noise.name, {})
                .get(Keys.PROBABILITY, "no default")
            )
            if default_probability == "no default":
                assert config_probability == row_noise.probability
            else:
                assert config_probability == default_probability
        for col in form.columns:
            for noise_type in col.noise_types:
                config_level = config[form.name].column_noise[col.name][noise_type.name]
                # FIXME: Is there a way to allow for adding new keys when they
                #  don't exist in baseline? eg the for/if loops below depend on their
                #  being row_noise, token_noise, and additional parameters at the
                #  baseline level ('noise_type in col.noise_types')
                #  Would we ever want to allow for adding non-baseline default noise?
                if noise_type.probability:
                    config_probability = config_level[Keys.PROBABILITY]
                    default_probability = (
                        DEFAULT_NOISE_VALUES.get(form.name, {})
                        .get(Keys.COLUMN_NOISE, {})
                        .get(col.name, {})
                        .get(noise_type.name, {})
                        .get(Keys.PROBABILITY, "no default")
                    )
                    if default_probability == "no default":
                        assert config_probability == noise_type.probability
                    else:
                        assert config_probability == default_probability
                if noise_type.additional_parameters:
                    config_additional_parameters = {
                        k: v
                        for k, v in config_level.to_dict().items()
                        if k != Keys.PROBABILITY
                    }
                    default_additional_parameters = (
                        DEFAULT_NOISE_VALUES.get(form.name, {})
                        .get(Keys.COLUMN_NOISE, {})
                        .get(col.name, {})
                        .get(noise_type.name, {})
                    )
                    default_additional_parameters = {
                        k: v
                        for k, v in default_additional_parameters.items()
                        if k != Keys.PROBABILITY
                    }
                    if default_additional_parameters == {}:
                        assert (
                            config_additional_parameters == noise_type.additional_parameters
                        )
                    else:
                        # Confirm config includes default values
                        for key, value in default_additional_parameters.items():
                            assert config_additional_parameters[key] == value
                        # Check that non-default values are baseline
                        baseline_keys = [
                            k
                            for k in config_additional_parameters
                            if k not in default_additional_parameters
                        ]
                        for key, value in config_additional_parameters.items():
                            if key not in baseline_keys:
                                continue
                            assert noise_type.additional_parameters[key] == value


def test_get_configuration_with_user_override(mocker):
    """Tests that the default configuration get updated when a user configuration is supplied."""
    mock = mocker.patch("pseudopeople.configuration.generator.ConfigTree")
    config = {
        FORMS.census.name: {
            Keys.ROW_NOISE: {NOISE_TYPES.omission.name: {Keys.PROBABILITY: 0.05}},
            Keys.COLUMN_NOISE: {
                "first_name": {NOISE_TYPES.typographic.name: {Keys.PROBABILITY: 0.05}}
            },
        }
    }
    _ = get_configuration(config)
    mock.assert_called_once_with(layers=["baseline", "default", "user"])
    update_calls = [
        call
        for call in mock.mock_calls
        if ".update({" in str(call) and "layer='user'" in str(call)
    ]
    assert len(update_calls) == 1


def test_loading_from_yaml(tmp_path):
    user_config = {
        FORMS.census.name: {
            Keys.COLUMN_NOISE: {
                COLUMNS.age.name: {
                    NOISE_TYPES.age_miswriting.name: {
                        Keys.PROBABILITY: 0.5,
                    },
                },
            },
        },
    }
    filepath = tmp_path / "user_dict.yaml"
    with open(filepath, "w") as file:
        yaml.dump(user_config, file)

    default_config = get_configuration()[FORMS.census.name][Keys.COLUMN_NOISE][
        COLUMNS.age.name
    ][NOISE_TYPES.age_miswriting.name].to_dict()
    updated_config = get_configuration(filepath)[FORMS.census.name][Keys.COLUMN_NOISE][
        COLUMNS.age.name
    ][NOISE_TYPES.age_miswriting.name].to_dict()

    assert (
        default_config[Keys.POSSIBLE_AGE_DIFFERENCES]
        == updated_config[Keys.POSSIBLE_AGE_DIFFERENCES]
    )
    # check that 1 got replaced with 0 probability
    assert updated_config[Keys.PROBABILITY] == 0.5


@pytest.mark.parametrize(
    "user_config, expected",
    [
        ([-2, -1, 2], {-2: 1 / 3, -1: 1 / 3, 1: 0, 2: 1 / 3}),
        ({-2: 0.3, 1: 0.5, 2: 0.2}, {-2: 0.3, -1: 0, 1: 0.5, 2: 0.2}),
    ],
    ids=["list", "dict"],
)
def test_format_miswrite_ages(user_config, expected):
    """Test that user-supplied dictionary properly updates ConfigTree object.
    This includes zero-ing out default values that don't exist in the user config
    """
    user_config = {
        FORMS.census.name: {
            Keys.COLUMN_NOISE: {
                COLUMNS.age.name: {
                    NOISE_TYPES.age_miswriting.name: {
                        Keys.POSSIBLE_AGE_DIFFERENCES: user_config,
                    },
                },
            },
        },
    }

    config = get_configuration(user_config)[FORMS.census.name][Keys.COLUMN_NOISE][
        COLUMNS.age.name
    ][NOISE_TYPES.age_miswriting.name][Keys.POSSIBLE_AGE_DIFFERENCES].to_dict()

    assert config == expected


@pytest.mark.parametrize(
    "config, match",
    [
        ({"fake_form": {}}, "Invalid form '.*' provided. Valid forms are "),
        (
            {FORMS.acs.name: {"other_noise": {}}},
            "Invalid configuration key '.*' provided for form '.*'. ",
        ),
        (
            {FORMS.acs.name: {Keys.ROW_NOISE: {"fake_noise": {}}}},
            "Invalid noise type '.*' provided for form '.*'. ",
        ),
        (
            {FORMS.acs.name: {Keys.ROW_NOISE: {NOISE_TYPES.omission.name: {"fake": {}}}}},
            "Invalid parameter '.*' provided for form '.*' and noise type '.*'. ",
        ),
        (
            {FORMS.acs.name: {Keys.COLUMN_NOISE: {"fake_column": {}}}},
            "Invalid column '.*' provided for form '.*'. ",
        ),
        (
            {FORMS.acs.name: {Keys.COLUMN_NOISE: {COLUMNS.age.name: {"fake_noise": {}}}}},
            "Invalid noise type '.*' provided for form '.*' for column '.*'. ",
        ),
        (
            {
                FORMS.acs.name: {
                    Keys.COLUMN_NOISE: {
                        COLUMNS.age.name: {NOISE_TYPES.missing_data.name: {"fake": 1}}
                    }
                }
            },
            "Invalid parameter '.*' provided for form '.*' for column '.*' and noise type '.*'",
        ),
    ],
    ids=[
        "invalid form",
        "invalid noise type",
        "invalid row noise type",
        "Invalid row noise type parameter",
        "Invalid column",
        "Invalid column noise type",
        "Invalid column noise type parameter",
    ],
)
def test_overriding_nonexistent_keys_fails(config, match):
    with pytest.raises(ValueError, match=match):
        get_configuration(config)


@pytest.mark.parametrize(
    "value, error, match",
    [
        (-0.01, ValueError, "must be between 0.0 and 1.0"),
        (1.01, ValueError, "must be between 0.0 and 1.0"),
        (2, TypeError, "must be a float"),
        ("a", TypeError, "must be a float"),
    ],
)
def test_validate_standard_parameters_failures(value, error, match):
    """Test that a runtime error is thrown if a user provides bad standard
    probability values

    NOTE: This also includes cell_probability values and technically any
    other values not provided a unique validation function.
    """
    with pytest.raises(error, match=match):
        get_configuration(
            {
                FORMS.census.name: {
                    Keys.COLUMN_NOISE: {
                        COLUMNS.age.name: {
                            NOISE_TYPES.age_miswriting.name: {
                                Keys.PROBABILITY: value,
                            },
                        },
                    },
                },
            },
        )


@pytest.mark.parametrize(
    "perturbations, error, match",
    [
        (-1, TypeError, "Invalid configuration type"),
        ([-1, 0.4, 1], TypeError, "must be ints"),
        ({-1: 0.5, 0.4: 0.2, 1: 0.3}, TypeError, "must be ints"),
        ([-1, 0, 1], ValueError, "Cannot include 0"),
        ({-1: 0.5, 4: 0.2, 0: 0.3}, ValueError, "Cannot include 0"),
        ({-1: 0.1, 1: 1}, TypeError, "must be floats"),
        ({-1: 0.1, 1: 0.8}, ValueError, "must sum to 1"),
    ],
    ids=[
        "bad type",
        "non-int keys list",
        "non-int keys dict",
        "include 0 list",
        "include 0 dict",
        "non-float values",
        "not sum to 1",
    ],
)
def test_validate_miswrite_ages_failures(perturbations, error, match):
    """Test that a runtime error is thrown if the user provides bad possible_age_differences"""
    with pytest.raises(error, match=match):
        get_configuration(
            {
                FORMS.census.name: {
                    Keys.COLUMN_NOISE: {
                        COLUMNS.age.name: {
                            NOISE_TYPES.age_miswriting.name: {
                                Keys.PROBABILITY: 1,
                                Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                            },
                        },
                    },
                },
            },
        )


@pytest.mark.parametrize(
    "probabilities, error, match",
    [
        (0.2, TypeError, "Invalid configuration type"),
        ([0.5, 0.5, 0.5, 0.5], ValueError, "must be 5"),
        ([1, 0.0, 0.0, 0.0, 0.0], TypeError, "must be floats"),
        ([0.2, 0.2, "foo", 0.2, 0.2], TypeError, "must be floats"),
        ([-0.1, 0.2, 0.2, 0.2, 0.2], ValueError, "must be between 0.0 and 1.0"),
        ([1.01, 0.2, 0.2, 0.2, 0.2], ValueError, "must be between 0.0 and 1.0"),
    ],
)
def test_validate_miswrite_zipcode_digit_probabilities_failures(probabilities, error, match):
    """Test that a runtime error is thrown if the user provides bad zipcode_digit_probabilities"""
    with pytest.raises(error, match=match):
        get_configuration(
            {
                FORMS.census.name: {
                    Keys.COLUMN_NOISE: {
                        COLUMNS.zipcode.name: {
                            NOISE_TYPES.zipcode_miswriting.name: {
                                Keys.CELL_PROBABILITY: 1,
                                Keys.ZIPCODE_DIGIT_PROBABILITIES: probabilities,
                            },
                        },
                    },
                },
            },
        )

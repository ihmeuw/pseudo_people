from pathlib import Path
from typing import Dict, Union

import yaml
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.configuration.validator import validate_user_configuration
from pseudopeople.constants.data_values import DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASETS

# Define non-baseline default items
# NOTE: default values are defined in entity_types.RowNoiseType and entity_types.ColumnNoiseType
DEFAULT_NOISE_VALUES = {
    DATASETS.census.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.do_not_respond.name: {
                Keys.ROW_PROBABILITY: DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[
                    DATASETS.census.name
                ],
            }
        },
    },
    DATASETS.acs.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.do_not_respond.name: {
                Keys.ROW_PROBABILITY: DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[
                    DATASETS.acs.name
                ],
            },
        },
    },
    DATASETS.cps.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.do_not_respond.name: {
                Keys.ROW_PROBABILITY: DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[
                    DATASETS.cps.name
                ],
            },
        },
    },
    DATASETS.tax_w2_1099.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omit_row.name: {
                Keys.ROW_PROBABILITY: 0.005,
            },
        },
        Keys.COLUMN_NOISE: {
            COLUMNS.ssn.name: {
                NOISE_TYPES.copy_from_household_member.name: {
                    Keys.CELL_PROBABILITY: 0.00,
                }
            },
        },
    },
    # No noise of any kind for SSN in the SSA observer
    DATASETS.ssa.name: {
        Keys.COLUMN_NOISE: {
            COLUMNS.ssn.name: {
                noise_type.name: {
                    Keys.CELL_PROBABILITY: 0.0,
                }
                for noise_type in COLUMNS.ssn.noise_types
            },
        },
    },
}


def get_configuration(user_configuration: Union[Path, str, Dict] = None) -> ConfigTree:
    """
    Gets a noising configuration ConfigTree, optionally overridden by a user-provided YAML.

    :param user_configuration: A path to the YAML file or a dictionary defining user overrides for the defaults
    :return: a ConfigTree object of the noising configuration
    """

    if type(user_configuration) == str:
        if user_configuration.lower() == Keys.NO_NOISE:
            config_type = Keys.NO_NOISE
            user_configuration = None
    else:
        config_type = Keys.DEFAULT
    noising_configuration = _generate_configuration(config_type)
    if user_configuration:
        add_user_configuration(noising_configuration, user_configuration)

    return noising_configuration


def _generate_configuration(config_type: str) -> ConfigTree:
    default_config_layers = [
        "baseline",
        "default",
        "user",
    ]
    noising_configuration = ConfigTree(layers=default_config_layers)
    # Instantiate the configuration file with baseline values
    baseline_dict = {}
    # Loop through each dataset
    for dataset in DATASETS:
        dataset_dict = {}
        row_noise_dict = {}
        column_dict = {}

        # Loop through row noise types
        for row_noise in dataset.row_noise_types:
            row_noise_type_dict = {}
            if row_noise.row_probability is not None:
                if config_type == Keys.NO_NOISE:
                    noise_level = 0.0
                else:
                    noise_level = row_noise.row_probability
                row_noise_type_dict[Keys.ROW_PROBABILITY] = noise_level
            if row_noise_type_dict:
                row_noise_dict[row_noise.name] = row_noise_type_dict

        # Loop through columns and their applicable column noise types
        for column in dataset.columns:
            column_noise_dict = {}
            for noise_type in column.noise_types:
                column_noise_type_dict = {}
                if noise_type.cell_probability is not None:
                    if config_type == Keys.NO_NOISE:
                        noise_level = 0.0
                    else:
                        noise_level = noise_type.cell_probability
                    column_noise_type_dict[Keys.CELL_PROBABILITY] = noise_level
                if noise_type.additional_parameters is not None:
                    for key, value in noise_type.additional_parameters.items():
                        column_noise_type_dict[key] = value
                if column_noise_type_dict:
                    column_noise_dict[noise_type.name] = column_noise_type_dict
            if column_noise_dict:
                column_dict[column.name] = column_noise_dict

        # Compile
        if row_noise_dict:
            dataset_dict[Keys.ROW_NOISE] = row_noise_dict
        if column_dict:
            dataset_dict[Keys.COLUMN_NOISE] = column_dict

        # Add the dataset's dictionary to baseline
        if dataset_dict:
            baseline_dict[dataset.name] = dataset_dict

    noising_configuration.update(baseline_dict, layer="baseline")

    # Update configuration with non-baseline default values
    if config_type == Keys.DEFAULT:
        noising_configuration.update(DEFAULT_NOISE_VALUES, layer="default")
    return noising_configuration


def add_user_configuration(
    noising_configuration: ConfigTree, user_configuration: Union[Path, str, Dict]
) -> None:
    if isinstance(user_configuration, (Path, str)):
        with open(user_configuration, "r") as f:
            user_configuration = yaml.full_load(f)

    validate_user_configuration(user_configuration, noising_configuration)

    user_configuration = _format_user_configuration(noising_configuration, user_configuration)
    noising_configuration.update(user_configuration, layer="user")


def _format_user_configuration(default_config: ConfigTree, user_dict: Dict) -> Dict:
    """Formats the user's configuration file as necessary, so it can properly
    update noising configuration to be used
    """
    user_dict = _format_misreport_age_perturbations(default_config, user_dict)
    return user_dict


def _format_misreport_age_perturbations(default_config: ConfigTree, user_dict: Dict) -> Dict:
    # Format any age perturbation lists as a dictionary with uniform probabilities
    for dataset in user_dict:
        user_perturbations = (
            user_dict[dataset]
            .get(Keys.COLUMN_NOISE, {})
            .get("age", {})
            .get(NOISE_TYPES.misreport_age.name, {})
            .get(Keys.POSSIBLE_AGE_DIFFERENCES, {})
        )
        if not user_perturbations:
            continue
        formatted = {}
        default_perturbations = default_config[dataset][Keys.COLUMN_NOISE]["age"][
            NOISE_TYPES.misreport_age.name
        ][Keys.POSSIBLE_AGE_DIFFERENCES]
        # Replace default configuration with 0 probabilities
        for perturbation in default_perturbations:
            formatted[perturbation] = 0
        if isinstance(user_perturbations, list):
            # Add user perturbations with uniform probabilities
            uniform_prob = 1 / len(user_perturbations)
            for perturbation in user_perturbations:
                formatted[perturbation] = uniform_prob
        else:
            for perturbation, prob in user_perturbations.items():
                formatted[perturbation] = prob

        user_dict[dataset][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.misreport_age.name][
            Keys.POSSIBLE_AGE_DIFFERENCES
        ] = formatted

    return user_dict

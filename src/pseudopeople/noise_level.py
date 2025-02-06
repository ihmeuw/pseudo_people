from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pseudopeople.configuration import Keys
from pseudopeople.constants import data_values
from pseudopeople.constants.metadata import DatasetNames

if TYPE_CHECKING:
    from pseudopeople.configuration.noise_configuration import NoiseConfiguration
    from pseudopeople.dataset import Dataset


def _get_census_omission_noise_levels(
    population: pd.DataFrame,
    base_probability: float = data_values.DO_NOT_RESPOND_BASE_PROBABILITY,
) -> pd.Series[float]:
    """
    Helper function for do_not_respond noising based on demography of age, race/ethnicity, and sex.

    :param population: a dataframe containing records of simulants
    :param base_probability: base probability for do_not_respond
    :return: a pd.Series of probabilities
    """
    probabilities = pd.Series(base_probability, index=population.index)
    probabilities += (
        population["race_ethnicity"]
        .astype(str)
        .map(data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_RACE)
    )
    ages = pd.Series(np.arange(population["age"].max() + 1))
    for sex in ["Female", "Male"]:
        effect_by_age_bin = data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE[sex]
        # NOTE: calling pd.cut on a large array with an IntervalIndex is slow,
        # see https://github.com/pandas-dev/pandas/issues/47614
        # Instead, we only pd.cut the unique ages, then do a simpler `.map` on the age column
        age_bins = pd.cut(ages, bins=effect_by_age_bin.index)
        effect_by_age = pd.Series(
            age_bins.map(effect_by_age_bin),
            index=ages,
        )
        sex_mask = population["sex"] == sex
        probabilities[sex_mask] += (
            population[sex_mask]["age"].map(effect_by_age).astype(float)
        )
    probabilities[probabilities < 0.0] = 0.0
    probabilities[probabilities > 1.0] = 1.0
    return probabilities


def get_apply_do_not_respond_noise_level(
    configuration: NoiseConfiguration, dataset: Dataset, noise_type: str
) -> pd.Series[float]:
    dataset_name = dataset.dataset_schema.name
    noise_levels = _get_census_omission_noise_levels(dataset.data)

    # Apply an overall non-response rate of 27.6% for Current Population Survey (CPS)
    if dataset_name == DatasetNames.CPS:
        noise_levels += 0.276

    # Apply user-configured noise level
    configured_noise_level: float = configuration.get_row_probability(
        dataset.dataset_schema.name, "do_not_respond"
    )
    default_noise_level = data_values.DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[dataset_name]
    noise_levels = noise_levels * (configured_noise_level / default_noise_level)

    # Account for ACS and CPS oversampling
    if dataset_name in [DatasetNames.ACS, DatasetNames.CPS]:
        noise_levels = 0.5 + noise_levels / 2

    return noise_levels

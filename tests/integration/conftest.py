from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from _pytest.legacypath import TempdirFactory

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.dataset import Dataset
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
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from pseudopeople.utilities import coerce_dtypes

ROW_PROBABILITY = 0.05
CELL_PROBABILITY = 0.25
SEED = 0
STATE = "RI"

# TODO: Replace this with the record ID column when implemented (MIC-4039)
IDX_COLS = {
    DATASET_SCHEMAS.census.name: [COLUMNS.simulant_id.name, COLUMNS.year.name],
    DATASET_SCHEMAS.acs.name: [COLUMNS.simulant_id.name, COLUMNS.survey_date.name],
    DATASET_SCHEMAS.cps.name: [COLUMNS.simulant_id.name, COLUMNS.survey_date.name],
    DATASET_SCHEMAS.wic.name: [COLUMNS.simulant_id.name, COLUMNS.year.name],
    DATASET_SCHEMAS.ssa.name: [COLUMNS.simulant_id.name, COLUMNS.ssa_event_type.name],
    DATASET_SCHEMAS.tax_w2_1099.name: [
        COLUMNS.simulant_id.name,
        COLUMNS.tax_year.name,
        COLUMNS.employer_id.name,
    ],
    DATASET_SCHEMAS.tax_1040.name: [
        COLUMNS.simulant_id.name,
        COLUMNS.tax_year.name,
    ],
}


@pytest.fixture(scope="session")
def split_sample_data_dir(tmpdir_factory: TempdirFactory) -> Path:
    datasets = [
        DatasetNames.CENSUS,
        DatasetNames.ACS,
        DatasetNames.CPS,
        DatasetNames.SSA,
        DatasetNames.TAXES_W2_1099,
        DatasetNames.WIC,
        DatasetNames.TAXES_1040,
    ]
    split_sample_data_dir = tmpdir_factory.mktemp("split_sample_data")
    for dataset_name in datasets:
        data_path = paths.SAMPLE_DATA_ROOT / dataset_name / f"{dataset_name}.parquet"
        data = pd.read_parquet(data_path)
        # Split the sample dataset into two and save in tmpdir_factory
        # We are spliting on household_id as a solution for how to keep households together
        # for the tax 1040 dataset.
        outdir = split_sample_data_dir.mkdir(dataset_name)
        if dataset_name in [
            DatasetNames.TAXES_1040,
            DatasetNames.TAXES_W2_1099,
        ]:
            hh_ids = data.loc[
                data[COLUMNS.tax_year.name] == 2020, COLUMNS.household_id.name
            ].unique()
            hh1_ids = hh_ids[: int(len(hh_ids) / 2)]
            split_household_mask = data[COLUMNS.household_id.name].isin(hh1_ids)
            data[split_household_mask].to_parquet(outdir / f"{dataset_name}_1.parquet")
            data[~split_household_mask].to_parquet(outdir / f"{dataset_name}_2.parquet")
        else:
            split_idx = int(len(data) / 2)
            data[:split_idx].to_parquet(outdir / f"{dataset_name}_1.parquet")
            data[split_idx:].to_parquet(outdir / f"{dataset_name}_2.parquet")

    return Path(split_sample_data_dir)


@pytest.fixture(scope="session")
def split_sample_data_dir_state_edit(
    tmpdir_factory: TempdirFactory, split_sample_data_dir: Path
) -> Path:
    # This replaces our old tmpdir fixture we were using because this more accurately
    # represents our sample data directory structure with a subdirectory for each
    # dataset and storing all files in the fixture.
    datasets = [
        DatasetNames.CENSUS,
        DatasetNames.ACS,
        DatasetNames.CPS,
        DatasetNames.SSA,
        DatasetNames.TAXES_W2_1099,
        DatasetNames.WIC,
        DatasetNames.TAXES_1040,
    ]
    split_sample_data_dir_state_edit = tmpdir_factory.mktemp("split_sample_data_state_edit")
    for dataset_name in datasets:
        outdir = split_sample_data_dir_state_edit.mkdir(dataset_name)
        data_paths = [
            split_sample_data_dir / dataset_name / f"{dataset_name}_1.parquet",
            split_sample_data_dir / dataset_name / f"{dataset_name}_2.parquet",
        ]
        for data_path in data_paths:
            data = pd.read_parquet(data_path)
            # We do not filter by state for SSA
            if dataset_name != DatasetNames.SSA:
                # Add a state so we can filter for integration tests
                state_column = [column for column in data.columns if "state" in column]
                data.loc[data.reset_index().index % 2 == 0, state_column] = STATE
                data.to_parquet(outdir / data_path.name)

    return Path(split_sample_data_dir_state_edit)


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    """Returns a custom configuration dict to be used in noising"""
    config = get_configuration().to_dict()  # default config

    # Increase row noise probabilities to 5% and column cell_probabilities to 25%
    for dataset_name in config:
        dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
        config[dataset_schema.name][Keys.ROW_NOISE] = {
            noise_type.name: {
                Keys.ROW_PROBABILITY: ROW_PROBABILITY,
            }
            for noise_type in dataset_schema.row_noise_types
            if noise_type != NOISE_TYPES.duplicate_with_guardian
        }
        for col in [c for c in dataset_schema.columns if c.noise_types]:
            config[dataset_name][Keys.COLUMN_NOISE][col.name] = {
                noise_type.name: {
                    Keys.CELL_PROBABILITY: CELL_PROBABILITY,
                }
                for noise_type in col.noise_types
            }

    # FIXME: Remove when record_id is added as the truth deck for datasets.
    # For integration tests, we will NOT duplicate rows with guardian duplication.
    # This is because we want to be able to compare the noised and unnoised data
    # and a big assumption we make is that simulant_id and household_id are the
    # truth decks in our datasets.
    config[DATASET_SCHEMAS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.duplicate_with_guardian.name
    ] = {
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.0,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0.0,
    }
    # Update SSA dataset to noise 'ssn' but NOT noise 'ssa_event_type' since that
    # will be used as an identifier along with simulant_id
    # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
    config[DATASET_SCHEMAS.ssa.name][Keys.COLUMN_NOISE][COLUMNS.ssa_event_type.name] = {
        noise_type.name: {
            Keys.CELL_PROBABILITY: 0,
        }
        for noise_type in COLUMNS.ssa_event_type.noise_types
    }
    return config


# Noised sample datasets
@pytest.fixture(scope="module")
def noised_sample_data_decennial_census(config: dict[str, Any]) -> pd.DataFrame:
    return generate_decennial_census(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_american_community_survey(config: dict[str, Any]) -> pd.DataFrame:
    return generate_american_community_survey(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_current_population_survey(config: dict[str, Any]) -> pd.DataFrame:
    return generate_current_population_survey(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_women_infants_and_children(config: dict[str, Any]) -> pd.DataFrame:
    return generate_women_infants_and_children(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_social_security(config: dict[str, Any]) -> pd.DataFrame:
    return generate_social_security(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_taxes_w2_and_1099(config: dict[str, Any]) -> pd.DataFrame:
    return generate_taxes_w2_and_1099(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_taxes_1040(config: dict[str, Any]) -> pd.DataFrame:
    return generate_taxes_1040(seed=SEED, year=None, config=config)


####################
# HELPER FUNCTIONS #
####################


def get_unnoised_data(dataset_name: str) -> Dataset:
    result = _initialize_dataset_with_sample(dataset_name)
    result.data = coerce_dtypes(result.data, result.dataset_schema)
    return result


def _initialize_dataset_with_sample(dataset_name: str) -> Dataset:
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    data_path = paths.SAMPLE_DATA_ROOT / dataset_name / f"{dataset_name}.parquet"
    dataset = Dataset(dataset_schema, pd.read_parquet(data_path), SEED)

    return dataset


def _get_common_datasets(
    unnoised_dataset: Dataset, noised_dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index[int]]:
    """Use unique columns to determine shared non-NA rows between noised and
    unnoised data. Note that we cannot use the original index because that
    gets reset after noising, i.e. the unique columns must NOT be noised.
    """
    idx_cols = IDX_COLS.get(unnoised_dataset.dataset_schema.name)
    unnoised_dataset._reformat_dates_for_noising()
    unnoised_dataset.data = coerce_dtypes(
        unnoised_dataset.data, unnoised_dataset.dataset_schema
    )
    check_original = unnoised_dataset.data.set_index(idx_cols)
    check_noised = noised_dataset.set_index(idx_cols)
    # Ensure the idx_cols are unique
    assert check_original.index.duplicated().sum() == 0
    assert check_noised.index.duplicated().sum() == 0
    shared_idx = pd.Index(set(check_original.index).intersection(set(check_noised.index)))
    check_original = check_original.loc[shared_idx]
    check_noised = check_noised.loc[shared_idx]
    return check_noised, check_original, shared_idx

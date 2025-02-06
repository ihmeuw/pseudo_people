from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from layered_config_tree import LayeredConfigTree
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.entity_types import ColumnNoiseType, NoiseType, RowNoiseType
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
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS, Column
from pseudopeople.utilities import (
    count_number_of_tokens_per_string,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
)
from tests.integration.conftest import (
    CELL_PROBABILITY,
    IDX_COLS,
    SEED,
    STATE,
    _get_common_datasets,
    _initialize_dataset_with_sample,
    get_unnoised_data,
)
from tests.unit.test_configuration import COLUMN_NOISE_TYPES

DATASET_GENERATION_FUNCS: dict[str, Callable[..., Any]] = {
    DATASET_SCHEMAS.census.name: generate_decennial_census,
    DATASET_SCHEMAS.acs.name: generate_american_community_survey,
    DATASET_SCHEMAS.cps.name: generate_current_population_survey,
    DATASET_SCHEMAS.ssa.name: generate_social_security,
    DATASET_SCHEMAS.tax_w2_1099.name: generate_taxes_w2_and_1099,
    DATASET_SCHEMAS.wic.name: generate_women_infants_and_children,
    DATASET_SCHEMAS.tax_1040.name: generate_taxes_1040,
}

TOKENS_PER_STRING_MAPPER: dict[str, Callable[..., pd.Series[int]]] = {
    NOISE_TYPES.make_ocr_errors.name: partial(
        count_number_of_tokens_per_string, pd.Series(load_ocr_errors().index)
    ),
    NOISE_TYPES.make_phonetic_errors.name: partial(
        count_number_of_tokens_per_string,
        pd.Series(load_phonetic_errors().index),
    ),
    NOISE_TYPES.write_wrong_digits.name: lambda x: x.astype(str)
    .str.replace(r"[^\d]", "", regex=True)
    .str.len(),
    NOISE_TYPES.make_typos.name: partial(
        count_number_of_tokens_per_string, pd.Series(load_qwerty_errors_data().index)
    ),
}


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_generate_dataset_from_multiple_shards(
    dataset_name: str,
    engine: str,
    config: dict[str, Any],
    request: FixtureRequest,
    split_sample_data_dir: Path,
    mocker: MockerFixture,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that the amount of noising is approximately the same whether we
    noise a single sample dataset or we concatenate and noise multiple datasets
    """
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    original = _initialize_dataset_with_sample(dataset_name)
    noised_sample = request.getfixturevalue(f"noised_sample_data_{dataset_name}")

    noised_dataset = generation_function(
        seed=SEED,
        year=None,
        source=split_sample_data_dir,
        engine=engine,
        config=config,
    )

    if engine == "dask":
        noised_dataset = noised_dataset.compute()

    # Check same order of magnitude of rows was removed -- we don't know the
    # full data size (we would need unnoised data for that), so we just check
    # for similar lengths
    assert 0.9 <= (len(noised_dataset) / len(noised_sample)) <= 1.1
    # Check that columns are identical
    assert noised_dataset.columns.equals(noised_sample.columns)

    # Check that each columns level of noising are similar
    check_noised_dataset, check_original_dataset, shared_dataset_idx = _get_common_datasets(
        original, noised_dataset
    )

    config_tree = get_configuration(config)
    for col_name in check_noised_dataset.columns:
        col = COLUMNS.get_column(col_name)
        if col.noise_types:
            noise_level_dataset, to_compare_dataset_idx = _get_column_noise_level(
                column=col,
                noised_data=check_noised_dataset,
                unnoised_data=check_original_dataset,
                common_idx=shared_dataset_idx,
            )

            # Validate noise for each data object
            _validate_column_noise_level(
                dataset_name=dataset_name,
                check_data=check_original_dataset,
                check_idx=to_compare_dataset_idx,
                noise_level=noise_level_dataset,
                col=col,
                config=config_tree,
                fuzzy_name="test_generate_dataset_from_sample_and_source_dataset",
                validator=fuzzy_checker,
            )


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_seed_behavior(
    dataset_name: str, engine: str, config: dict[str, Any], request: FixtureRequest
) -> None:
    """Tests seed behavior"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    original = get_unnoised_data(dataset_name)
    if engine == "dask":
        noised_data = generation_function(
            seed=SEED,
            year=None,
            config=config,
            engine=engine,
        )
    else:
        noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    # Generate new (non-fixture) noised datasets with the same seed and a different
    # seed as the fixture
    noised_data_same_seed = generation_function(
        seed=SEED,
        year=None,
        config=config,
        engine=engine,
    )
    noised_data_different_seed = generation_function(
        seed=SEED + 1,
        year=None,
        config=config,
        engine=engine,
    )

    if engine == "dask":
        noised_data = noised_data.compute()
        noised_data_same_seed = noised_data_same_seed.compute()
        noised_data_different_seed = noised_data_different_seed.compute()

    assert not original.data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_column_dtypes(
    dataset_name: str, engine: str, config: dict[str, Any], request: FixtureRequest
) -> None:
    """Tests that column dtypes are as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)

    if engine == "dask":
        generation_function = DATASET_GENERATION_FUNCS[dataset_name]
        noised_data = generation_function(
            seed=SEED,
            year=None,
            config=config,
            engine=engine,
        ).compute()
    else:
        noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")

    for col_name in noised_data.columns:
        col = COLUMNS.get_column(col_name)
        expected_dtype = col.dtype_name
        if expected_dtype == np.dtype(object):
            # str dtype is 'object'
            # Check that they are actually strings and not some other
            # type of object.
            actual_types = noised_data[col.name].dropna().apply(type)
            assert (actual_types == str).all(), actual_types.unique()
        assert noised_data[col.name].dtype == expected_dtype


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_column_noising(
    dataset_name: str,
    engine: str,
    config: dict[str, Any],
    request: FixtureRequest,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that columns are noised as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    original = _initialize_dataset_with_sample(dataset_name)
    if engine == "dask":
        generation_function = DATASET_GENERATION_FUNCS[dataset_name]
        noised_data = generation_function(
            seed=SEED,
            year=None,
            config=config,
            engine=engine,
        ).compute()
    else:
        noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original, shared_idx = _get_common_datasets(original, noised_data)

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
            assert different_check.any()

            noise_level = different_check.sum()

            # Validate column noise level
            _validate_column_noise_level(
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


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_row_noising_omit_row_or_do_not_respond(
    dataset_name: str, engine: str, config: dict[str, Any], request: FixtureRequest
) -> None:
    """Tests that omit_row and do_not_respond row noising are being applied"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    idx_cols = IDX_COLS.get(dataset_name)
    original = get_unnoised_data(dataset_name)
    original_data = original.data.set_index(idx_cols)
    if engine == "dask":
        generation_function = DATASET_GENERATION_FUNCS[dataset_name]
        noised_data = generation_function(
            seed=SEED,
            year=None,
            config=config,
            engine=engine,
        ).compute()
    else:
        noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    noised_data = noised_data.set_index(idx_cols)
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


@pytest.mark.skip(reason="TODO: Implement duplication row noising")
@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
def test_row_noising_duplication(dataset_name: str) -> None:
    """Tests that duplication row noising is being applied"""
    ...


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_generate_dataset_with_year(dataset_name: str, engine: str) -> None:
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    year = 2030  # not default 2020
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    original = get_unnoised_data(dataset_name)
    # Generate a new (non-fixture) noised dataset for a single year
    noised_data = generation_function(year=year, engine=engine)
    if engine == "dask":
        noised_data = noised_data.compute()
    assert not original.data.equals(noised_data)


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_dataset_filter_by_year(
    mocker: MockerFixture, dataset_name: str, engine: str
) -> None:
    """Mock the noising function so that it returns the date column of interest
    with the original (unnoised) values to ensure filtering is happening
    """
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    year = 2030  # not default 2020

    # Generate a new (non-fixture) noised dataset for a single year but mocked such
    # that no noise actually happens (otherwise the years would get noised and
    # we couldn't tell if the filter was working properly)
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = generation_function(year=year, engine=engine)
    if engine == "dask":
        noised_data = noised_data.compute()
    dataset = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    assert (noised_data[dataset.date_column_name] == year).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_dataset_filter_by_year_with_full_dates(
    mocker: MockerFixture, dataset_name: str, engine: str
) -> None:
    """Mock the noising function so that it returns the date column of interest
    with the original (unnoised) values to ensure filtering is happening
    """
    year = 2030  # not default 2020
    # Generate a new (non-fixture) noised dataset for a single year but mocked such
    # that no noise actually happens (otherwise the years would get noised and
    # we couldn't tell if the filter was working properly)
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = generation_function(year=year, engine=engine)
    if engine == "dask":
        noised_data = noised_data.compute()
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)

    noised_column = noised_data[dataset_schema.date_column_name]
    if is_datetime(noised_column):
        years = noised_column.dt.year
    else:
        years = pd.to_datetime(noised_column, format=dataset_schema.date_format).dt.year

    if dataset_schema == DATASET_SCHEMAS.ssa:
        assert (years <= year).all()
    else:
        assert (years == year).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_generate_dataset_with_state_filtered(
    dataset_name: str,
    engine: str,
    split_sample_data_dir_state_edit: Path,
    mocker: MockerFixture,
) -> None:
    """Test that values returned by dataset generators are only for the specified state"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]

    # Skip noising (noising can incorrect select another state)
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    noised_data = generation_function(
        source=split_sample_data_dir_state_edit, state=STATE, engine=engine
    )
    if engine == "dask":
        noised_data = noised_data.compute()

    assert (noised_data[dataset_schema.state_column_name] == STATE).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_generate_dataset_with_state_unfiltered(
    dataset_name: str,
    engine: str,
    split_sample_data_dir_state_edit: Path,
    mocker: MockerFixture,
) -> None:
    # Important note: Currently the way this test is working is we have a fixture where we have
    # edited the sample data so half of it has a state to filter to. However, when we split the
    # sample data and do this, all the 2020 data (the year we default to for all generate_xxx functions)
    # results in the 2020 data being only in one of the files. In practice, this is how we want
    # the functionality of these functions to work but we should consider updating fixtures/tests
    # in the future. - albrja
    """Test that values returned by dataset generators are for all locations if state unspecified"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)

    # Skip noising (noising can incorrect select another state)
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = generation_function(source=split_sample_data_dir_state_edit, engine=engine)

    assert len(noised_data[dataset_schema.state_column_name].unique()) > 1


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_dataset_filter_by_state_and_year(
    mocker: MockerFixture,
    split_sample_data_dir_state_edit: Path,
    dataset_name: str,
    engine: str,
) -> None:
    """Test that dataset generation works with state and year filters in conjunction"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = generation_function(
        source=split_sample_data_dir_state_edit,
        year=year,
        state=STATE,
        engine=engine,
    )
    if engine == "dask":
        noised_data = noised_data.compute()
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    assert (noised_data[dataset_schema.date_column_name] == year).all()
    assert (noised_data[dataset_schema.state_column_name] == STATE).all()


@pytest.mark.parametrize(
    "dataset_name",
    [DATASET_SCHEMAS.acs.name, DATASET_SCHEMAS.cps.name],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_dataset_filter_by_state_and_year_with_full_dates(
    mocker: MockerFixture,
    split_sample_data_dir_state_edit: Path,
    dataset_name: str,
    engine: str,
) -> None:
    """Test that dataset generation works with state and year filters in conjunction"""
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = generation_function(
        source=split_sample_data_dir_state_edit,
        year=year,
        state=STATE,
        engine=engine,
    )
    if engine == "dask":
        noised_data = noised_data.compute()
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)

    noised_column = noised_data[dataset_schema.date_column_name]
    if is_datetime(noised_column):
        years = noised_column.dt.year
    else:
        years = pd.to_datetime(noised_column, format=dataset_schema.date_format).dt.year

    assert (years == year).all()
    assert (noised_data[dataset_schema.state_column_name] == STATE).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_generate_dataset_with_bad_state(
    dataset_name: str,
    engine: str,
    split_sample_data_dir_state_edit: Path,
    mocker: MockerFixture,
) -> None:
    """Test that bad state values result in informative ValueErrors"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    bad_state = "Silly State That Doesn't Exist"
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    with pytest.raises(ValueError, match=bad_state.upper()):
        df = generation_function(
            source=split_sample_data_dir_state_edit,
            state=bad_state,
            engine=engine,
        )
        if engine == "dask":
            df.compute()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "dask",
    ],
)
def test_generate_dataset_with_bad_year(
    dataset_name: str, engine: str, split_sample_data_dir: Path, mocker: MockerFixture
) -> None:
    """Test that a ValueError is raised both for a bad year and a year that has no data"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    bad_year = 0
    no_data_year = 2000
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    with pytest.raises(ValueError):
        df = generation_function(
            source=split_sample_data_dir,
            year=bad_year,
            engine=engine,
        )
        if engine == "dask":
            df.compute()
    with pytest.raises(ValueError):
        df = generation_function(
            source=split_sample_data_dir,
            year=no_data_year,
            engine=engine,
        )
        if engine == "dask":
            df.compute()


####################
# HELPER FUNCTIONS #
####################


def _validate_column_noise_level(
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
    # Fuzzy checker
    validator.fuzzy_assert_proportion(
        name=fuzzy_name,
        observed_numerator=noise_level,
        observed_denominator=len(check_data.loc[check_idx, col.name]),
        target_proportion=expected_noise,
        name_additional=f"{dataset_name}_{col.name}_{col_noise_type.name}",
    )


def _get_column_noise_level(
    column: Column,
    noised_data: pd.DataFrame,
    unnoised_data: pd.DataFrame,
    common_idx: pd.Index[int],
) -> tuple[int, pd.Index[int]]:

    # Check that originally missing data remained missing
    originally_missing_sample_idx = unnoised_data.index[unnoised_data[column.name].isna()]

    assert noised_data.loc[originally_missing_sample_idx, column.name].isna().all()

    # Check for noising where applicable
    to_compare_sample_idx = common_idx.difference(originally_missing_sample_idx)
    different_check: npt.NDArray[np.bool_] = np.array(
        unnoised_data.loc[to_compare_sample_idx, column.name].values
        != noised_data.loc[to_compare_sample_idx, column.name].values
    )

    return different_check.sum(), to_compare_sample_idx

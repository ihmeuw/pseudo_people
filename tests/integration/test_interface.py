from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS, Column
from pseudopeople.utilities import coerce_dtypes
from tests.constants import DATASET_GENERATION_FUNCS
from tests.integration.conftest import (
    IDX_COLS,
    SEED,
    STATE,
    _get_common_datasets,
    get_unnoised_data,
)
from tests.utilities import (
    initialize_dataset_with_sample,
    run_column_noising_tests,
    run_omit_row_or_do_not_respond_tests,
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
def test_noising_sharded_vs_unsharded_data(
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

    unnoised_dataset = initialize_dataset_with_sample(dataset_name)
    single_shard_noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    multi_shard_noised_data = generation_function(
        seed=SEED,
        year=None,
        source=split_sample_data_dir,
        engine=engine,
        config=config,
    )
    if engine == "dask":
        multi_shard_noised_data = multi_shard_noised_data.compute()

    assert multi_shard_noised_data.columns.equals(single_shard_noised_data.columns)

    # This index handling is adapted from _get_common_datasets
    # in integration/conftest.py
    # Define indexes
    idx_cols = IDX_COLS.get(unnoised_dataset.dataset_schema.name)
    unnoised_dataset._reformat_dates_for_noising()
    unnoised_dataset.data = coerce_dtypes(
        unnoised_dataset.data, unnoised_dataset.dataset_schema
    )
    check_original = unnoised_dataset.data.set_index(idx_cols)
    check_single_noised = single_shard_noised_data.set_index(idx_cols)
    check_multi_noised = multi_shard_noised_data.set_index(idx_cols)

    # Ensure the idx_cols are unique
    assert check_original.index.duplicated().sum() == 0
    assert check_single_noised.index.duplicated().sum() == 0
    assert check_multi_noised.index.duplicated().sum() == 0

    # Get shared indexes
    shared_idx = pd.Index(
        set(check_original.index)
        .intersection(set(check_single_noised.index))
        .intersection(set(check_multi_noised.index))
    )
    check_original = check_original.loc[shared_idx]
    check_single_noised = check_single_noised.loc[shared_idx]
    check_multi_noised = check_multi_noised.loc[shared_idx]

    for col in check_single_noised.columns:
        fuzzy_checker.fuzzy_assert_proportion(
            target_proportion=(check_single_noised[col] != check_original[col]).mean(),
            observed_numerator=(check_multi_noised[col] != check_original[col]).sum(),
            observed_denominator=len(check_original),
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

    # Generate a new (non-fixture) dataset for a single year but mocked such
    # that no noise actually happens (otherwise the years would get noised and
    # we couldn't tell if the filter was working properly)
    mocker.patch("pseudopeople.dataset.Dataset._noise_dataset")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    data = generation_function(year=year, engine=engine)
    if engine == "dask":
        data = data.compute()
    dataset = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    assert (data[dataset.date_column_name] == year).all()


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

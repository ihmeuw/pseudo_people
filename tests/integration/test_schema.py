import pytest
from _pytest.fixtures import FixtureRequest

from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from tests.integration.conftest import (
    _get_common_datasets,
    _initialize_dataset_with_sample,
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
def test_unnoised_id_cols(dataset_name: str, request: FixtureRequest) -> None:
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    unnoised_id_cols = [COLUMNS.simulant_id.name]
    if dataset_name != DATASET_SCHEMAS.ssa.name:
        unnoised_id_cols.append(COLUMNS.household_id.name)
    original = _initialize_dataset_with_sample(dataset_name)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original, _ = _get_common_datasets(original, noised_data)
    assert (
        (
            check_original.reset_index()[unnoised_id_cols]
            == check_noised.reset_index()[unnoised_id_cols]
        )
        .all()
        .all()
    )

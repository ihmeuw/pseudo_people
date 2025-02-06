import pytest
from layered_config_tree import LayeredConfigTree

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.entity_types import RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS
from tests.integration.conftest import _initialize_dataset_with_sample


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
def test_dataset_missingness(dataset_name: str) -> None:
    """Tests that missingness is accurate with dataset.data."""
    dataset = _initialize_dataset_with_sample(dataset_name)
    # We must manually clean the data for noising since we are recreating our main noising loop
    dataset._clean_input_data()
    dataset._reformat_dates_for_noising()
    config = get_configuration()
    # NOTE: This is recreating Dataset._noise_dataset but adding assertions for missingness
    for noise_type in NOISE_TYPES:
        if isinstance(noise_type, RowNoiseType):
            if config.has_noise_type(dataset.dataset_schema.name, noise_type.name):
                noise_type(dataset, config)
                # Check missingness is synced with data
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
        else:
            for column in dataset.data.columns:
                if config.has_noise_type(
                    dataset.dataset_schema.name, noise_type.name, column
                ):
                    noise_type(dataset, config, column)
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.noise import noise_dataset
from pseudopeople.schema_entities import COLUMNS, DATASETS, Dataset
from pseudopeople.utilities import configure_logging_to_terminal, get_state_abbreviation


def _generate_dataset(
    dataset: Dataset,
    source: Union[Path, str],
    seed: int,
    config: Union[Path, str, Dict],
    user_filters: List[tuple],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Helper for generating noised datasets.

    :param dataset:
        Dataset needing to be noised
    :param source:
        Root directory of data input which needs to be noised
    :param seed:
        Seed for controlling randomness
    :param config:
        Object to configure noise levels
    :param user_filters:
        List of parquet filters, possibly empty
    :param verbose:
        Log with verbosity if True. Default is False.
    :return:
        Noised dataset data in a pd.DataFrame
    """
    configure_logging_to_terminal(verbose)
    configuration_tree = get_configuration(config)

    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    data_paths = fetch_filepaths(dataset, source)
    if not data_paths:
        raise DataSourceError(
            f"No datasets found at directory {str(source)}. "
            "Please provide the path to the unmodified root data directory."
        )
    validate_data_path_suffix(data_paths)
    noised_dataset = []
    iterator = (
        tqdm(data_paths, desc="Noising data", leave=False)
        if len(data_paths) > 1
        else data_paths
    )

    for data_path in iterator:
        logger.debug(f"Loading data from {data_path}.")
        data = _load_data_from_path(data_path, user_filters)
        if data.empty:
            continue
        data = _reformat_dates_for_noising(data, dataset)
        data = _coerce_dtypes(data, dataset)
        noised_data = noise_dataset(dataset, data, configuration_tree, seed)
        noised_data = _extract_columns(dataset.columns, noised_data)
        noised_dataset.append(noised_data)

    noised_dataset = pd.concat(noised_dataset, ignore_index=True)

    # Known pandas bug: pd.concat does not preserve category dtypes so we coerce
    # again after concat (https://github.com/pandas-dev/pandas/issues/51362)
    noised_dataset = _coerce_dtypes(noised_dataset, dataset)

    logger.debug("*** Finished ***")

    return noised_dataset


def _coerce_dtypes(data: pd.DataFrame, dataset: Dataset):
    # Coerce dtypes prior to noising to catch issues early as well as
    # get most columns away from dtype 'category' and into 'object' (strings)
    for col in dataset.columns:
        if col.dtype_name != data[col.name].dtype.name:
            data[col.name] = data[col.name].astype(col.dtype_name)
    return data


def _load_data_from_path(data_path: Union[Path, dict], user_filters: List[Tuple]):
    """Load data from a data file given a data_path and a year_filter."""
    if isinstance(data_path, dict):
        data = load_and_prep_1040_data(data_path, user_filters)
    else:
        data = load_file(data_path, user_filters)
    return data


def _reformat_dates_for_noising(data: pd.DataFrame, dataset: Dataset):
    """Formats SSA event_date and dates of birth, so they can be noised."""
    data = data.copy()
    if COLUMNS.ssa_event_date.name in data.columns and dataset == DATASETS.ssa:
        # event_date -> YYYYMMDD
        data[COLUMNS.ssa_event_date.name] = data[COLUMNS.ssa_event_date.name].dt.strftime(
            "%Y%m%d"
        )
    if COLUMNS.dob.name in data.columns:
        # date_of_birth -> MM/DD/YYYY
        data[COLUMNS.dob.name] = data[COLUMNS.dob.name].dt.strftime("%m/%d/%Y")
    return data


def _extract_columns(columns_to_keep, noised_dataset):
    """Helper function for test mocking purposes"""
    if columns_to_keep:
        noised_dataset = noised_dataset[[c.name for c in columns_to_keep]]
    return noised_dataset


def generate_decennial_census(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople decennial census dataset which represents simulated
    responses to the US Census Bureau's Census of Population and Housing.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The year (format YYYY) to include in the dataset. Must be a decennial
        year (e.g. 2020, 2030, 2040). Will return an empty pd.DataFrame if there are no
        data with this year. If None is provided, data from all years are
        included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty pd.DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :return: A pd.DataFrame of simulated decennial census data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.census.date_column_name, "==", year))
    if state:
        user_filters.append(
            (DATASETS.census.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(DATASETS.census, source, seed, config, user_filters, verbose)


def generate_american_community_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople ACS dataset which represents simulated responses to
    the ACS survey.

    The American Community Survey (ACS) is an ongoing household survey conducted by
    the US Census Bureau that gathers information on a rolling basis about
    American community populations. Information collected includes ancestry,
    citizenship, education, income, language proficiency, migration, employment,
    disability, and housing characteristics.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The survey date year (format YYYY) to include in the dataset. Will
        return an empty pd.DataFrame if there are no data with this year. If None is
        provided, data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty pd.DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :return: A pd.DataFrame of simulated ACS data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.extend(
            [
                (DATASETS.acs.date_column_name, ">=", pd.Timestamp(f"{year}-01-01")),
                (DATASETS.acs.date_column_name, "<=", pd.Timestamp(f"{year}-12-31")),
            ]
        )
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.acs.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(DATASETS.acs, source, seed, config, user_filters, verbose)


def generate_current_population_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople CPS dataset which represents simulated responses to
    the CPS survey.

    The Current Population Survey (CPS) is a household survey conducted by the
    US Census Bureau and the US Bureau of Labor Statistics. This survey is administered
    by Census Bureau field representatives across the country through both personal
    and telephone interviews. CPS collects labor force data, such as annual work
    activity and income, veteran status, school enrollment, contingent employment,
    worker displacement, job tenure, and more.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The survey date year (format YYYY) to include in the dataset. Will
        return an empty pd.DataFrame if there are no data with this year. If None is
        provided, data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty pd.DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :return: A pd.DataFrame of simulated CPS data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.extend(
            [
                (DATASETS.cps.date_column_name, ">=", pd.Timestamp(f"{year}-01-01")),
                (DATASETS.cps.date_column_name, "<=", pd.Timestamp(f"{year}-12-31")),
            ]
        )
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.cps.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(DATASETS.cps, source, seed, config, user_filters, verbose)


def generate_taxes_w2_and_1099(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople W2 and 1099 tax dataset which represents simulated
    tax form data.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The tax year (format YYYY) to include in the dataset. Will return
        an empty pd.DataFrame if there are no data with this year. If None is provided,
        data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty pd.DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :return: A pd.DataFrame of simulated W2 and 1099 tax data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.tax_w2_1099.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.tax_w2_1099.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.tax_w2_1099, source, seed, config, user_filters, verbose
    )


def generate_women_infants_and_children(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople WIC dataset which represents a simulated version of
    the administrative data that would be recorded by WIC. This is a yearly file
    of information about all simulants enrolled in the program as of the end of that year.

    The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC)
    is a government benefits program designed to support mothers and young children.
    The main qualifications are income and the presence of young children in the home.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The year (format YYYY) to include in the dataset. Will return an
        empty pd.DataFrame if there are no data with this year. If None is provided,
        data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty pd.DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :return: A pd.DataFrame of simulated WIC data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.wic.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.wic.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(DATASETS.wic, source, seed, config, user_filters, verbose)


def generate_social_security(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople SSA dataset which represents simulated Social Security
    Administration (SSA) data.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The latest year (format YYYY) to include in the dataset; will also
        include all previous years. Will return an empty pd.DataFrame if there are no
        data on or before this year. If None is provided, data from all years are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :return: A pd.DataFrame of simulated SSA data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append(
            (DATASETS.ssa.date_column_name, "<=", pd.Timestamp(f"{year}-12-31"))
        )
        seed = seed * 10_000 + year
    return _generate_dataset(DATASETS.ssa, source, seed, config, user_filters, verbose)


def fetch_filepaths(dataset, source):
    # todo: add typing
    if dataset.name == DatasetNames.TAXES_1040:
        tax_dataset_names = [
            DatasetNames.TAXES_1040,
            DatasetNames.TAXES_W2_1099,
            DatasetNames.TAXES_DEPENDENTS,
        ]
        for tax_dataset in tax_dataset_names:
            directory = source / tax_dataset
            dataset_paths = [x for x in directory.glob(f"{tax_dataset}*")]
            sorted_dataset_paths = sorted(dataset_paths)
            if tax_dataset == DatasetNames.TAXES_1040:
                data_paths = [{} for i in range(len(sorted_dataset_paths))]
            for i in range(len(sorted_dataset_paths)):
                data_paths[i][tax_dataset] = sorted_dataset_paths[i]
    else:
        source = Path(source) / dataset.name
        data_paths = [x for x in source.glob(f"{dataset.name}*")]

    return data_paths


def load_and_prep_1040_data(data_path: dict, user_filters: List[Tuple]) -> pd.DataFrame:
    # Function that loads all tax datasets and formats them to the DATASET.1040 schema
    return pd.DataFrame()


def validate_data_path_suffix(data_paths):
    if isinstance(data_paths[0], dict):
        suffix = set(x.suffix for item in data_paths for x in list(item.values()))
    else:
        suffix = set(x.suffix for x in data_paths)
    if len(suffix) > 1:
        raise DataSourceError(
            f"Only one type of file extension expected but more than one found: {suffix}. "
            "Please provide the path to the unmodified root data directory."
        )

    return None


def load_file(data_path: Path, user_filters: List[Tuple]) -> pd.DataFrame:
    if data_path.suffix == ".parquet":
        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(data_path, filters=user_filters).to_pandas()
    else:
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {data_path.suffix}"
        )
    if not isinstance(data, pd.DataFrame):
        raise DataSourceError(
            f"File located at {data_path} must contain a pandas DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )

    return data

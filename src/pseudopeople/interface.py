from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from packaging.version import parse
from tqdm import tqdm

from pseudopeople import __version__ as psp_version
from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.dataset import DatasetData
from pseudopeople.exceptions import DataSourceError
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS, DatasetSchema
from pseudopeople.utilities import (
    coerce_dtypes,
    configure_logging_to_terminal,
    get_state_abbreviation,
)


def _generate_dataset(
    dataset_schema: DatasetSchema,
    source: Union[Path, str],
    seed: int,
    config: Union[Path, str, Dict],
    user_filters: List[tuple],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Helper for generating noised datasets.

    :param dataset_schema:
        Schema object for dataset that will be noised
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
    configuration_tree = get_configuration(config, dataset_schema, user_filters)

    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    else:
        source = Path(source)
        validate_source_compatibility(source, dataset_schema)

    data_paths = fetch_filepaths(dataset_schema, source)
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

    for data_path_index, data_path in enumerate(iterator):
        logger.debug(f"Loading data from {data_path}.")
        dataset_data = DatasetData(
            dataset_schema, data_path, user_filters, f"{seed}_{data_path_index}"
        )

        if not dataset_data:
            continue

        noised_data = dataset_data.get_noised_data(configuration_tree, NOISE_TYPES)
        noised_dataset.append(noised_data)

    # Check if all shards for the dataset are empty
    if len(noised_dataset) == 0:
        raise ValueError(
            "Invalid value provided for 'state' or 'year'. No data found with "
            f"the user provided 'state' or 'year' filters at {data_path}."
        )
    noised_dataset = pd.concat(noised_dataset, ignore_index=True)

    # Known pandas bug: pd.concat does not preserve category dtypes so we coerce
    # again after concat (https://github.com/pandas-dev/pandas/issues/51362)
    noised_dataset = coerce_dtypes(
        noised_dataset,
        dataset_schema,
        cleanse_int_cols=True,
    )

    logger.debug("*** Finished ***")

    return noised_dataset


def validate_source_compatibility(source: Path, dataset_schema: DatasetSchema):
    # TODO [MIC-4546]: Clean this up w/ metadata and update test_interface.py tests to be generic
    directories = [x.name for x in source.iterdir() if x.is_dir()]
    if dataset_schema.name not in directories:
        raise FileNotFoundError(
            f"Could not find '{dataset_schema.name}' in '{source}'. Please check that the provided source "
            "directory is correct. If using the sample data, no source is required. If providing a source, "
            f"a directory should provided that has a subdirectory for '{dataset_schema.name}'. "
        )
    changelog = source / "CHANGELOG.rst"
    if changelog.exists():
        version = _get_data_changelog_version(changelog)
        if version > parse("1.4.2"):
            raise DataSourceError(
                f"The provided simulated population data is incompatible with this version of pseudopeople ({psp_version}).\n"
                "A newer version of simulated population data has been provided.\n"
                "Please upgrade the pseudopeople package."
            )
        if version < parse("1.4.2"):
            raise DataSourceError(
                f"The provided simulated population data is incompatible with this version of pseudopeople ({psp_version}).\n"
                "The simulated population data has been corrupted.\n"
                "Please re-download the simulated population data."
            )
    else:
        raise DataSourceError(
            f"The provided simulated population data is incompatible with this version of pseudopeople ({psp_version}).\n"
            "An older version of simulated population data has been provided.\n"
            "Please either request updated simulated population data or downgrade the pseudopeople package."
        )


def _get_data_changelog_version(changelog):
    with open(changelog, "r") as file:
        first_line = file.readline()
    version = parse(first_line.split("**")[1].split("-")[0].strip())
    return version


def generate_decennial_census(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople decennial census dataset which represents
    simulated responses to the US Census Bureau's Census of Population
    and Housing.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate a simulated decennial census of
        the simulated population (format YYYY, e.g., 2030). Must be a
        decennial year (e.g., 2020, 2030, 2040). Default is 2020. If
        `None` is passed instead, data for all available years are
        included in the returned dataset.

    :param state:

        The US state for which to generate a simulated census of the
        simulated population, or `None` (default) to generate data for
        all available US states. The returned dataset will contain data
        for simulants living in the specified state on Census Day (April
        1) of the specified year. Can be a full state name or a state
        abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :return:

        A `pandas.DataFrame` of simulated decennial census data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASET_SCHEMAS.census.date_column_name, "==", year))
    if state is not None:
        user_filters.append(
            (DATASET_SCHEMAS.census.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASET_SCHEMAS.census, source, seed, config, user_filters, verbose
    )


def generate_american_community_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople ACS dataset which represents simulated
    responses to the ACS survey.

    The American Community Survey (ACS) is an ongoing household survey
    conducted by the US Census Bureau that gathers information on a
    rolling basis about American community populations. Information
    collected includes ancestry, citizenship, education, income,
    language proficiency, migration, employment, disability, and housing
    characteristics.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate simulated American Community
        Surveys of the simulated population (format YYYY, e.g., 2036);
        the simulated dataset will contain records for surveys conducted
        on any date in the specified year. Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param state:

        The US state for which to generate simulated American Community
        Surveys of the simulated population, or `None` (default) to
        generate data for all available US states. The returned dataset
        will contain survey data for simulants living in the specified
        state during the specified year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :return:

        A `pandas.DataFrame` of simulated ACS data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        try:
            user_filters.extend(
                [
                    (
                        DATASET_SCHEMAS.acs.date_column_name,
                        ">=",
                        pd.Timestamp(year=year, month=1, day=1),
                    ),
                    (
                        DATASET_SCHEMAS.acs.date_column_name,
                        "<=",
                        pd.Timestamp(year=year, month=12, day=31),
                    ),
                ]
            )
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.extend(
            [(DATASET_SCHEMAS.acs.state_column_name, "==", get_state_abbreviation(state))]
        )
    return _generate_dataset(DATASET_SCHEMAS.acs, source, seed, config, user_filters, verbose)


def generate_current_population_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople CPS dataset which represents simulated
    responses to the CPS survey.

    The Current Population Survey (CPS) is a household survey conducted
    by the US Census Bureau and the US Bureau of Labor Statistics. This
    survey is administered by Census Bureau field representatives across
    the country through both personal and telephone interviews. CPS
    collects labor force data, such as annual work activity and income,
    veteran status, school enrollment, contingent employment, worker
    displacement, job tenure, and more.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate simulated Current Population
        Surveys of the simulated population (format YYYY, e.g., 2036);
        the simulated dataset will contain records for surveys conducted
        on any date in the specified year. Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param state:

        The US state for which to generate simulated Current Population
        Surveys of the simulated population, or `None` (default) to
        generate data for all available US states. The returned dataset
        will contain survey data for simulants living in the specified
        state during the specified year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :return:

        A `pandas.DataFrame` of simulated CPS data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        try:
            user_filters.extend(
                [
                    (
                        DATASET_SCHEMAS.cps.date_column_name,
                        ">=",
                        pd.Timestamp(year=year, month=1, day=1),
                    ),
                    (
                        DATASET_SCHEMAS.cps.date_column_name,
                        "<=",
                        pd.Timestamp(year=year, month=12, day=31),
                    ),
                ]
            )
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.extend(
            [(DATASET_SCHEMAS.cps.state_column_name, "==", get_state_abbreviation(state))]
        )
    return _generate_dataset(DATASET_SCHEMAS.cps, source, seed, config, user_filters, verbose)


def generate_taxes_w2_and_1099(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople W2 and 1099 tax dataset which represents
    simulated tax form data.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The tax year for which to generate records (format YYYY, e.g.,
        2036); the simulated dataset will contain the W2 & 1099 tax
        forms filed by simulated employers for the specified year.
        Default is 2020. If `None` is passed instead, data for all
        available years are included in the returned dataset.

    :param state:

        The US state for which to generate tax records from the
        simulated population, or `None` (default) to generate data for
        all available US states. The returned dataset will contain W2 &
        1099 tax forms filed for simulants living in the specified state
        during the specified tax year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :return:

        A `pandas.DataFrame` of simulated W2 and 1099 tax data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASET_SCHEMAS.tax_w2_1099.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.append(
            (
                DATASET_SCHEMAS.tax_w2_1099.state_column_name,
                "==",
                get_state_abbreviation(state),
            )
        )
    return _generate_dataset(
        DATASET_SCHEMAS.tax_w2_1099, source, seed, config, user_filters, verbose
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
    Generates a pseudopeople WIC dataset which represents a simulated
    version of the administrative data that would be recorded by WIC.
    This is a yearly file of information about all simulants enrolled in
    the program as of the end of that year.

    The Special Supplemental Nutrition Program for Women, Infants, and
    Children (WIC) is a government benefits program designed to support
    mothers and young children. The main qualifications are income and
    the presence of young children in the home.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate WIC administrative records
        (format YYYY, e.g., 2036); the simulated dataset will contain
        records for simulants enrolled in WIC at the end of the
        specified year (or on May 1, 2041 if `year=2041` since that is
        the end date of the simulation). Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param state:

        The US state for which to generate WIC administrative records
        from the simulated population, or `None` (default) to generate
        data for all available US states. The returned dataset will
        contain records for enrolled simulants living in the specified
        state at the end of the specified year (or on May 1, 2041 if
        `year=2041` since that is the end date of the simulation). Can
        be a full state name or a state abbreviation (e.g., "Ohio" or
        "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.
    :return:

        A `pandas.DataFrame` of simulated WIC data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASET_SCHEMAS.wic.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.append(
            (DATASET_SCHEMAS.wic.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(DATASET_SCHEMAS.wic, source, seed, config, user_filters, verbose)


def generate_social_security(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople SSA dataset which represents simulated
    Social Security Administration (SSA) data.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The final year of simulated social security records to include
        in the dataset (format YYYY, e.g., 2036); will also include
        records from all previous years. Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :return:

        A `pandas.DataFrame` of simulated SSA data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or any prior years.
    """
    user_filters = []
    if year is not None:
        try:
            user_filters.append(
                (
                    DATASET_SCHEMAS.ssa.date_column_name,
                    "<=",
                    pd.Timestamp(year=year, month=12, day=31),
                )
            )
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")
        seed = seed * 10_000 + year
    return _generate_dataset(DATASET_SCHEMAS.ssa, source, seed, config, user_filters, verbose)


def generate_taxes_1040(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates a pseudopeople 1040 tax dataset which represents simulated
    tax form data.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The tax year for which to generate records (format YYYY, e.g.,
        2036); the simulated dataset will contain the 1040 tax forms
        filed by simulants for the specified year. Default is 2020. If
        `None` is passed instead, data for all available years are
        included in the returned dataset.

    :param state:

        The US state for which to generate tax records from the
        simulated population, or `None` (default) to generate data for
        all available US states. The returned dataset will contain 1040
        tax forms filed by simulants living in the specified state
        during the specified tax year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :return:

        A `pandas.DataFrame` of simulated 1040 tax data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASET_SCHEMAS.tax_1040.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.append(
            (DATASET_SCHEMAS.tax_1040.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASET_SCHEMAS.tax_1040, source, seed, config, user_filters, verbose
    )


def fetch_filepaths(dataset_schema: DatasetSchema, source: Path) -> Union[List, List[dict]]:
    # returns a list of filepaths for the dataset
    data_paths = get_dataset_filepaths(source, dataset_schema.name)

    return data_paths


def validate_data_path_suffix(data_paths) -> None:
    suffix = set(x.suffix for x in data_paths)
    if len(suffix) > 1:
        raise DataSourceError(
            f"Only one type of file extension expected but more than one found: {suffix}. "
            "Please provide the path to the unmodified root data directory."
        )

    return None


def get_dataset_filepaths(source: Path, dataset_name: str) -> List[Path]:
    directory = source / dataset_name
    dataset_paths = [x for x in directory.glob(f"{dataset_name}*")]
    sorted_dataset_paths = sorted(dataset_paths)
    return sorted_dataset_paths

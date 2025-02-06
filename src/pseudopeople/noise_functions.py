from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from pseudopeople.configuration import Keys
from pseudopeople.constants.noise_type_metadata import (
    COPY_HOUSEHOLD_MEMBER_COLS,
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
    HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP,
)
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.noise_scaling import (
    load_incorrect_select_options,
    load_nicknames_data,
)
from pseudopeople.utilities import (
    ensure_dtype,
    get_hash,
    get_index_to_noise,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
    two_d_array_choice,
    vectorized_choice,
)

if TYPE_CHECKING:
    from pseudopeople.configuration.noise_configuration import NoiseConfiguration
    from pseudopeople.dataset import Dataset


def omit_rows(
    dataset: Dataset, configuration: NoiseConfiguration, to_noise_index: pd.Index[int]
) -> None:
    """
    Function that omits rows from a dataset and returns only the remaining rows.  Note that for the ACS and CPS datasets
      we need to account for oversampling in the PRL simulation so a helper function has been added here to do so.
    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    """
    dataset.data = dataset.data.loc[dataset.data.index.difference(to_noise_index)]
    # TODO: Mic-4875 add update_missingness method to Dataset
    dataset.missingness = dataset.missingness.loc[
        dataset.data.index.difference(to_noise_index)
    ]


def apply_do_not_respond(
    dataset: Dataset, configuration: NoiseConfiguration, to_noise_index: pd.Index[int]
) -> None:
    """
    Applies targeted omission based on demographic model for census and surveys.

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    """
    # todo: this validation should be done much earlier
    required_columns = ("age", "race_ethnicity", "sex")
    missing_columns = [col for col in required_columns if col not in dataset.data.columns]
    if len(missing_columns):
        raise ValueError(
            f"Dataset {dataset.dataset_schema.name} is missing required columns: "
            f"{missing_columns}"
        )

    dataset.data = dataset.data.loc[dataset.data.index.difference(to_noise_index)]
    # TODO: Mic-4875 add update_missingness method to Dataset
    dataset.missingness = dataset.missingness.loc[
        dataset.data.index.difference(to_noise_index)
    ]
    # todo should we reset index here to get a RangeIndex?


# def duplicate_rows(
#     dataset: "Dataset",
#     configuration: LayeredConfigTree,
#     to_noise_index: pd.Index,
# ) -> None:
#     """

#     :param dataset:
#     :param configuration:
#     :param to_noise_index:
#     :return:
#     """
#     # todo actually duplicate rows
#     return dataset


def duplicate_with_guardian(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
) -> None:
    """
    Function that duplicates rows of a dataset. Rows that are duplicated fall into one of three groups of
    dependents that are typically duplicated in administrative data where a dependent lives in a different
    location than their guardian and they show up in the data at both locations. For simplicity, rows will
    be only duplicated once for a maximum of two rows per dependent. When a row is duplicated, one row will
    have the dependent's correct address and the other will have the guardian's address.
    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    """

    # Helper function to format group dataframe and merging with their dependents
    def _merge_dependents_and_guardians(
        dependents_df: pd.DataFrame, full_data: pd.DataFrame
    ) -> pd.DataFrame:
        # Merge dependents with their guardians. We have to merge twice to check
        # if either guardian is living at a separate location from the dependent.
        guardian_1s = full_data.loc[
            full_data["simulant_id"].isin(full_data["guardian_1"]),
            GUARDIAN_DUPLICATION_ADDRESS_COLUMNS + ["simulant_id"],
        ].add_prefix("guardian_1_")
        dependents_and_guardians_df = dependents_df.merge(
            guardian_1s,
            how="left",
            left_on=["guardian_1", "year"],
            right_on=["guardian_1_simulant_id", "guardian_1_year"],
        )
        del guardian_1s
        guardian_2s = full_data.loc[
            full_data["simulant_id"].isin(full_data["guardian_2"]),
            GUARDIAN_DUPLICATION_ADDRESS_COLUMNS + ["simulant_id"],
        ].add_prefix("guardian_2_")
        dependents_and_guardians_df = dependents_and_guardians_df.merge(
            guardian_2s,
            how="left",
            left_on=["guardian_2", "year"],
            right_on=["guardian_2_simulant_id", "guardian_2_year"],
        )
        del guardian_2s

        return dependents_and_guardians_df

    # Get dict of group type and formatted dataframe for that group that should be noised
    formatted_group_data = {}
    # Get dataframe for each dependent group to merge with guardians
    in_households_under_18 = dataset.data.loc[
        (dataset.data["age"] < 18)
        & (dataset.data["housing_type"] == "Household")
        & (dataset.data["guardian_1"].notna())
    ]
    in_college_under_24 = dataset.data.loc[
        (dataset.data["age"] < 24)
        & (dataset.data["housing_type"] == "College")
        & (dataset.data["guardian_1"].notna())
    ]

    # Merge dependents with their guardians
    formatted_group_data[
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18
    ] = _merge_dependents_and_guardians(in_households_under_18, dataset.data)
    formatted_group_data[
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24
    ] = _merge_dependents_and_guardians(in_college_under_24, dataset.data)
    # Note: We have two dicts (configuration and formatted_group_data) at this point that have
    # the key for the group and then a dataframe for that group or the group and the configured
    # noise level

    duplicated_rows = []
    for group, group_df in formatted_group_data.items():
        # Get index groups that can be noised based on dependent and guardian(s) addresses
        both_different_index = group_df.index[
            (group_df["household_id"] != group_df["guardian_1_household_id"])
            & (group_df["guardian_1_household_id"].notna())
            & (group_df["household_id"] != group_df["guardian_2_household_id"])
            & (group_df["guardian_2_household_id"].notna())
        ]
        # Choose which guardian to copy when dependent lives in different address from both guardians
        choices = pd.Series(
            dataset.randomness.choice(
                ["guardian_1", "guardian_2"],
                size=len(both_different_index),
            ),
            index=both_different_index,
        )
        group_df.loc[both_different_index, "copy_guardian"] = choices
        # Get remaining dependents that live in different address from one of their guardians
        guardian_1_different_index = group_df.index[
            (group_df["household_id"] != group_df["guardian_1_household_id"])
            & (group_df["guardian_1_household_id"].notna())
        ].difference(choices.index)
        group_df.loc[guardian_1_different_index, "copy_guardian"] = "guardian_1"
        guardian_2_different_index = group_df.index[
            (group_df["household_id"] != group_df["guardian_2_household_id"])
            & (group_df["guardian_2_household_id"].notna())
        ].difference(choices.index)
        group_df.loc[guardian_2_different_index, "copy_guardian"] = "guardian_2"

        # Noise data
        # TODO: Mic-4876 Can we only operate on the index eligible for noise and
        # not the entire dataset?
        noise_level: int | float = configuration.get_duplicate_with_guardian_probabilities(
            dataset.dataset_schema.name, parameter_name=group
        )
        to_noise_index = get_index_to_noise(
            dataset,
            noise_level,
        ).intersection(group_df.index)

        # Copy over address information from guardian to dependent
        for guardian in ["guardian_1", "guardian_2"]:
            index_to_copy = to_noise_index.intersection(
                group_df.index[group_df["copy_guardian"] == guardian]
            )
            # Don't try to copy if there are no rows to copy
            if index_to_copy.empty:
                continue
            noised_group_df = group_df.loc[index_to_copy]
            noised_group_df[GUARDIAN_DUPLICATION_ADDRESS_COLUMNS] = group_df.loc[
                index_to_copy,
                [f"{guardian}_" + column for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS],
            ]
            duplicated_rows.append(noised_group_df)

    if duplicated_rows:
        duplicated_rows_df: pd.DataFrame = pd.concat(duplicated_rows)
        # Update relationship to reference person for duplicated simulants based on housing type
        duplicated_rows_df["relationship_to_reference_person"] = duplicated_rows_df[
            "housing_type"
        ].map(HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP)

        # Clean columns
        duplicated_rows_df = duplicated_rows_df[dataset.data.columns]

        # Add duplicated rows to the original data and make sure that households
        # are grouped together by sorting by date and household_id
        # todo if this index is a RangeIndex, we can do concat with ignore_index=True
        index_start_value = dataset.data.index.max() + 1
        duplicated_rows_df.index = pd.Index(
            range(index_start_value, index_start_value + len(duplicated_rows_df))
        )
        # Note: This is where we would sort the data by year and household_id but
        # we ran into runtime issues. It may be sufficient to do it here since this
        # would be sorting at the shard level and not the entire dataset.
        data_with_duplicates = pd.concat([dataset.data, duplicated_rows_df])

        duplicated_rows_missing = dataset.is_missing(duplicated_rows_df)
        missingess_with_duplicates = pd.concat(
            [dataset.missingness, duplicated_rows_missing]
        ).reindex(data_with_duplicates.index)
        # Update
        dataset.data = data_with_duplicates.reset_index(drop=True)
        dataset.missingness = missingess_with_duplicates.reset_index(drop=True)


def choose_wrong_options(
    dataset: Dataset,
    _: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param dataset: Dataset object containing the data to be noised
    :param _: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    selection_type = {
        "employer_state": "state",
        "mailing_address_state": "state",
    }.get(str(column_name), column_name)

    selection_options = load_incorrect_select_options()

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]
    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(to_noise_index),
        random_generator=dataset.randomness,
    ).to_numpy()

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(new_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def copy_from_household_member(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that copies the value for a column from another household member.

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    copy_values = dataset.data.loc[to_noise_index, COPY_HOUSEHOLD_MEMBER_COLS[column_name]]
    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(copy_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def swap_months_and_days(
    dataset: Dataset,
    _: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that swaps month and day of dates.

    :param dataset: Dataset object containing the data to be noised
    :param _: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """
    from pseudopeople.schema_entities import DATEFORMATS

    date_format = dataset.dataset_schema.date_format

    to_swap_values = dataset.data.loc[to_noise_index, column_name]
    if date_format == DATEFORMATS.YYYYMMDD:  # YYYYMMDD
        year = to_swap_values.str[:4]
        month = to_swap_values.str[4:6]
        day = to_swap_values.str[6:]
        noised = year + day + month
    elif date_format == DATEFORMATS.MM_DD_YYYY:  # MM/DD/YYYY
        year = to_swap_values.str[6:]
        month = to_swap_values.str[:3]
        day = to_swap_values.str[3:6]
        noised = day + month + year
    elif date_format == DATEFORMATS.MMDDYYYY:  # MMDDYYYY
        year = to_swap_values.str[4:]
        month = to_swap_values.str[:2]
        day = to_swap_values.str[2:4]
        noised = day + month + year
    else:
        raise ValueError(f"Invalid date format in {dataset.dataset_schema.name}.")

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(noised, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def write_wrong_zipcode_digits(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that noises a 5 digit zipcode

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    to_noise_zipcodes = dataset.data.loc[to_noise_index, column_name]
    str_len = to_noise_zipcodes.str.len()
    if (str_len != 5).sum() > 0:
        # TODO: This is a BAD error message. It should never appear and if it
        #   does, the user shouldn't be checking the simulated population data.
        raise ValueError(
            "Zipcode data contains zipcodes that are not 5 digits long. Please check simulated population data."
        )

    shape = (len(to_noise_zipcodes), 5)

    # todo: Update when vectorized choice is improved
    possible_replacements = np.array(list("0123456789"))
    # Scale up noise levels to adjust for inclusive sampling with all numbers
    scaleup_factor = 1 / (1 - (1 / len(possible_replacements)))
    # Get configuration values for each piece of 5 digit zipcode
    digit_probabilities = scaleup_factor * np.array(
        configuration.get_value(
            dataset.dataset_schema.name,
            "write_wrong_zipcode_digits",
            Keys.ZIPCODE_DIGIT_PROBABILITIES,
            column_name,
        )
    )
    replace = dataset.randomness.random(shape) < digit_probabilities
    num_to_replace = replace.sum()
    random_digits = dataset.randomness.choice(possible_replacements, num_to_replace)

    # https://stackoverflow.com/a/9493192/
    # Changing this to a U5 numpy string type means that each string will have exactly 5 characters.
    # view("U1") then reinterprets this memory as an array of individual (Unicode) characters.
    same_len_col_exploded = to_noise_zipcodes.values.astype("U5").view("U1").reshape(shape)
    same_len_col_exploded[replace] = random_digits

    noised_values = same_len_col_exploded.view("U5").reshape(len(to_noise_zipcodes))
    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(noised_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def misreport_ages(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """Function to mis-write ages based on perturbation parameters included in
    the config file.

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    column = dataset.data.loc[to_noise_index, column_name]
    possible_perturbations: dict[int, float] = configuration.get_misreport_ages_probabilities(
        dataset.dataset_schema.name, column_name
    )

    perturbations = vectorized_choice(
        options=list(possible_perturbations.keys()),
        weights=list(possible_perturbations.values()),
        n_to_choose=len(to_noise_index),
        random_generator=dataset.randomness,
    )
    new_values = column.astype(int) + perturbations
    # Reflect negative values to positive
    new_values[new_values < 0] *= -1
    # If new age == original age, subtract 1
    new_values[new_values == column.astype(int)] -= 1

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(new_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def write_wrong_digits(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that noises numeric characters in a series.

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """
    column: pd.Series[str] = dataset.data.loc[to_noise_index, column_name]
    # This is a fix to not replacing the original token for noise options
    token_probability: float = configuration.get_token_probability(
        dataset.dataset_schema.name, "write_wrong_digits", column_name
    )
    token_noise_level = token_probability / 0.9

    max_str_length = column.str.len().max()

    possible_replacements = np.array(list("0123456789"))

    # https://stackoverflow.com/a/9493192/
    # Changing this to a numpy (not Python) string type means that it will have a fixed
    # number of characters, equal to the longest string in the array.
    # view("U1") then reinterprets this memory as an array of individual (Unicode) characters.
    same_len_col_exploded = (
        column.values.astype(str).view("U1").reshape((len(column), max_str_length))
    )
    # Surprisingly, Numpy does not provide a computationally efficient way to do
    # this check for which characters are eligible.
    # A head-to-head comparison found np.isin to be orders of magnitude slower than using Pandas here.
    # Also, np.isin fails silently with sets (see https://numpy.org/doc/stable/reference/generated/numpy.isin.html)
    # so be careful if testing that in the future!
    is_number = (
        pd.DataFrame(same_len_col_exploded).isin(set(possible_replacements)).to_numpy()
    )

    replace = np.zeros_like(is_number, dtype=bool)
    replace[is_number] = dataset.randomness.random(is_number.sum()) < token_noise_level
    num_to_replace = replace.sum()
    random_digits = dataset.randomness.choice(possible_replacements, num_to_replace)

    same_len_col_exploded[replace] = random_digits
    noised_column = same_len_col_exploded.view(f"U{max_str_length}").reshape(len(column))

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(noised_column, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def use_nicknames(
    dataset: Dataset,
    _: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that replaces a name with a choice of potential nicknames.

    :param dataset: Dataset object containing the data to be noised
    :param _: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """
    nicknames = load_nicknames_data()
    nickname_eligible_names = set(nicknames.index)
    column = dataset.data.loc[to_noise_index, column_name]
    # TODO: Mic-4876 This is just like guardian duplication, can we move finding an eligible index
    # outside of the noise function?
    have_nickname_idx = column.index[column.isin(nickname_eligible_names)]
    noised = two_d_array_choice(
        column.loc[have_nickname_idx],
        nicknames,
        dataset.randomness,
    )
    dataset.data.loc[have_nickname_idx, column_name] = ensure_dtype(
        pd.Series(noised, name=column_name, index=have_nickname_idx),
        dataset.data[column_name].dtype,
    )


def use_fake_names(
    dataset: Dataset,
    _: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that replaces a name with a fake name from a list of options.

    :param dataset: Dataset object containing the data to be noised
    :param _: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """
    fake_names = {
        "first_name": fake_first_names,
        "middle_name": fake_first_names,
        "last_name": fake_last_names,
        "spouse_first_name": fake_first_names,
        "spouse_last_name": fake_last_names,
        "dependent_1_first_name": fake_first_names,
        "dependent_1_last_name": fake_last_names,
        "dependent_2_first_name": fake_first_names,
        "dependent_2_last_name": fake_last_names,
        "dependent_3_first_name": fake_first_names,
        "dependent_3_last_name": fake_last_names,
        "dependent_4_first_name": fake_first_names,
        "dependent_4_last_name": fake_last_names,
    }
    options = fake_names[column_name]

    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(to_noise_index),
        random_generator=dataset.randomness,
    )

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(new_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def make_phonetic_errors(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that noises a string by replacing characters with phonetically similar characters.

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    # Load phonetic errors
    phonetic_errors = load_phonetic_errors()
    token_probability: float = configuration.get_token_probability(
        dataset.dataset_schema.name, "make_phonetic_errors", column_name
    )

    noised_values = _corrupt_tokens(
        phonetic_errors,
        dataset.data.loc[to_noise_index, column_name],
        token_probability,
        dataset.randomness,
    )

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(noised_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def leave_blanks(
    dataset: Dataset,
    _: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that takes a column and blanks out all values.

    :param dataset: Dataset object containing the data to be noised
    :param _: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(np.nan, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )
    dataset.missingness.loc[to_noise_index, column_name] = True


def make_typos(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """Function that applies noise to the string values
    representative of keyboard mistyping.

    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    qwerty_errors = load_qwerty_errors_data()
    qwerty_errors_eligible_chars = set(qwerty_errors.index)

    column = dataset.data.loc[to_noise_index, column_name]
    column = column.astype(str)
    token_noise_level = configuration.get_token_probability(
        dataset.dataset_schema.name, "make_typos", column_name
    )
    # TODO: remove this hard-coding
    include_token_probability_level = 0.1

    same_len_col_exploded = (
        # Somewhat counterintuitively, .astype(str) turns the column into a numpy,
        # fixed-length string type, U#, where # is the length of the longest string.
        column.values.astype(str)
        # Split into individual characters
        .view("U1").reshape((len(column), -1))
        # https://stackoverflow.com/a/9493192/
        # NOTE: Surprisingly, casting this to "object" (Python string type) seems to make
        # the rest of this method faster. Without this cast, you have to use np.char.add to concatenate,
        # which seems very slow for some reason?
        .astype("object")
    )

    # Surprisingly, Numpy does not provide a computationally efficient way to do
    # this check for which characters are eligible.
    # A head-to-head comparison found np.isin to be orders of magnitude slower than using Pandas here.
    # Also, np.isin fails silently with sets (see https://numpy.org/doc/stable/reference/generated/numpy.isin.html)
    # so be careful if testing that in the future!
    is_typo_option = (
        pd.DataFrame(same_len_col_exploded).isin(qwerty_errors_eligible_chars).to_numpy()
    )
    replace = np.zeros_like(is_typo_option)
    replace[is_typo_option] = (
        dataset.randomness.random(is_typo_option.sum()) < token_noise_level
    )
    keep_original = np.zeros_like(replace)
    keep_original[replace] = (
        dataset.randomness.random(replace.sum()) < include_token_probability_level
    )

    # Apply noising
    to_replace = same_len_col_exploded[replace]
    replace_random = dataset.randomness.random(replace.sum())
    number_of_options = qwerty_errors.count(axis=1)
    replace_option_index = np.floor(
        replace_random * number_of_options.loc[to_replace].to_numpy()
    )
    originals_to_keep = same_len_col_exploded[keep_original]

    qwerty_errors_stacked = qwerty_errors.stack()
    if not isinstance(qwerty_errors_stacked, pd.Series):
        raise ValueError("Your qwerty errors data had columns with multiple levels.")

    # Numpy does not have an efficient way to do this merge operation
    same_len_col_exploded[replace] = (
        pd.DataFrame({"to_replace": to_replace, "replace_option_index": replace_option_index})
        .reset_index()
        .merge(
            qwerty_errors_stacked.rename("replacement"),
            left_on=["to_replace", "replace_option_index"],
            right_index=True,
        )
        # Merge does not return the results in order
        .sort_values("index")
        .replacement.to_numpy()
    )

    same_len_col_exploded[keep_original] += originals_to_keep
    noised_column = np.sum(same_len_col_exploded, axis=1)

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(noised_column, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def make_ocr_errors(
    dataset: Dataset,
    configuration: NoiseConfiguration,
    to_noise_index: pd.Index[int],
    column_name: str,
) -> None:
    """
    Function that noises strings by replace characters with similar looking characters.
    :param dataset: Dataset object containing the data to be noised
    :param configuration: NoiseConfiguration object containing noise level values
    :param to_noise_index: pd.Index of rows to be noised
    :param column_name: String for column that will be noised
    """

    # Load OCR error dict
    ocr_errors = load_ocr_errors()
    token_probability: float = configuration.get_token_probability(
        dataset.dataset_schema.name, "make_ocr_errors", column_name
    )

    noised_values = _corrupt_tokens(
        ocr_errors,
        dataset.data.loc[to_noise_index, column_name],
        token_probability,
        dataset.randomness,
    )

    dataset.data.loc[to_noise_index, column_name] = ensure_dtype(
        pd.Series(noised_values, name=column_name, index=to_noise_index),
        dataset.data[column_name].dtype,
    )


def _corrupt_tokens(
    errors: pd.DataFrame,
    column: pd.Series[str],
    token_probability: float,
    random_generator: np.random.Generator,
) -> pd.Series[str]:
    """
    Performs token-level corruption on a string Series when the tokens to corrupt
    (and the tokens they get corrupted to) can be more than one character long.
    A token does not need to be the same length as a token it gets corrupted to.
    When both kinds of tokens are always one character, things are simpler,
    and a faster algorithm can be used; see typographic noise for an example of this.

    `errors` is a pandas DataFrame where the index contains tokens that are eligible to
    be corrupted, and the columns contain the corruptions that are possible.
    The column names do not matter.
    """
    # NOTE: This first section depends only on `errors` and could be cached -- we might
    # consider moving it into the load_* functions.
    # However, the amount of work here scales only with the size of `errors` and not
    # with the data.
    # It isn't possible to just make this a cached helper since `errors` is a DataFrame,
    # which is unhashable.
    max_token_length = errors.index.str.len().max()
    errors_eligible_tokens = (
        pd.Series(errors.index).groupby(errors.index.str.len()).apply(set).to_dict()
    )
    number_of_options = errors.count(axis=1)
    # Convert to 1D array for easy indexing
    errors_array = errors.values.reshape(-1)
    # Keep track of where in that 1D array the corruption options for each corruptable
    # token start
    errors_array_index_by_string = pd.Series(
        len(errors.columns) * np.array(range(len(errors))), index=errors.index
    )

    lengths: npt.NDArray[np.int_] = np.array(column.str.len().values)

    same_len_col_exploded = (
        column
        # Convert to numpy string dtype
        .values.astype(str)
        .view("U1")
        .reshape((len(column), lengths.max()))
    )

    # NOTE: Somewhat surprisingly, this seemed to perform better using Python string types than NumPy types.
    # Perhaps worth more investigation in the future.
    # Note that each item in result can be up to the number of characters of the *longest* corrupted token.
    result = np.empty(same_len_col_exploded.shape, dtype=str).astype("object")
    # Tracks the next character index where we have to pay attention to a given string.
    # When we corrupt a token, we skip to the end of that corrupted token before corrupting anything new,
    # so we don't have to consider that string again until we get to the end of it.
    next_due = np.zeros(len(column))
    # We proceed by character through all the strings at once
    for i in range(same_len_col_exploded.shape[1]):
        error_introduced = np.zeros(len(column), dtype=bool)
        assert np.all(next_due >= i)
        due = next_due == i

        # Longer tokens to-be-corrupted take precedence over shorter ones
        for token_length in range(max_token_length, 0, -1):
            if i + token_length > same_len_col_exploded.shape[1]:
                continue

            can_be_corrupted = ~error_introduced & due

            # Is the string already over at this index?
            # If so, we can ignore it.
            # NOTE: From here on, all the boolean arrays are implicitly
            # cumulative; that is, "long_enough" really means that it
            # is due, not already corrupted, *and* long enough. This allows us to short-circuit,
            # e.g. only check the length of strings that are due, and so on.
            long_enough = np.zeros(len(column), dtype=bool)
            long_enough[can_be_corrupted] = i + token_length <= lengths[can_be_corrupted]
            if long_enough.sum() == 0:
                continue
            del can_be_corrupted

            # Collect the tokens of the given length that start at this index.
            tokens = np.empty(len(column), dtype=f"U{token_length}")
            tokens[long_enough] = (
                same_len_col_exploded[long_enough, i : (i + token_length)]
                .view(f"U{token_length}")
                .reshape(long_enough.sum())
            )

            # Are these tokens that have corruptions?
            eligible = np.zeros(len(column), dtype=bool)
            eligible[long_enough] = (
                pd.Series(tokens[long_enough])
                .isin(errors_eligible_tokens.get(token_length, set()))
                .values
            )
            if eligible.sum() == 0:
                continue
            del long_enough

            # If so, should we corrupt them?
            corrupted = np.zeros(len(column), dtype=bool)
            corrupted[eligible] = random_generator.random(eligible.sum()) < token_probability
            if corrupted.sum() == 0:
                continue
            del eligible

            # If so, which corruption should we choose?
            # There is only a meaningful decision to make here when there are multiple
            # options for a given token, so we avoid drawing randomness when there is
            # only one.
            to_corrupt = tokens[corrupted]
            corrupted_token_index = np.zeros(len(to_corrupt), dtype=int)
            num_options = number_of_options.loc[to_corrupt].to_numpy()
            multiple_options = num_options > 1
            corrupted_token_index[multiple_options] = np.floor(
                random_generator.random(multiple_options.sum())
                * num_options[multiple_options]
            )
            # Get the actual string corresponding to the corrupted token chosen.
            # First, find the index in the errors array corresponding to the first corruption
            # of the token to be corrupted.
            # Then, add the offset of the corrupted token index.
            errors_array_indices = (
                errors_array_index_by_string.loc[to_corrupt].to_numpy()
                + corrupted_token_index
            )
            result[corrupted, i] = errors_array[errors_array_indices]
            # Will not be due again until we reach the end of this token
            next_due[corrupted] = i + token_length
            error_introduced[corrupted] = True

        # Strings that are not "due" -- that is, we already corrupted a source
        # token that included the current character index -- do not even need
        # the original character to be added to the result (it was *replaced*
        # by the corrupted token).
        use_original_char = ~error_introduced & due
        result[use_original_char, i] = same_len_col_exploded[use_original_char, i]
        next_due[use_original_char] = i + 1

    # "Un-explode" (re-concatenate) each string from its pieces.
    results: pd.Series[str] = pd.Series(result.sum(axis=1), index=column.index)
    return results

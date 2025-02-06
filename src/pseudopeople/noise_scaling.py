from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from pseudopeople.constants import paths
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS


def scale_choose_wrong_option(data: pd.DataFrame, column_name: str) -> float:
    """
    Function to scale noising for choose_wrong_option to adjust for the possibility
    of noising with the original values.
    """

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = get_options_for_column(column_name)

    # Scale to adjust for possibility of noising with original value
    noise_scaling_value = 1 / (1 - 1 / len(options))

    return noise_scaling_value


def scale_nicknames(data: pd.DataFrame, column_name: str) -> Any:
    # Constant calculated by number of names with nicknames / number of names used in PRL name mapping
    nicknames = load_nicknames_data()
    proportion_have_nickname: float = (
        data[column_name].isin(nicknames.index).sum() / data[column_name].notna().sum()
    )
    if proportion_have_nickname == 0.0:
        return 0.0
    return 1 / proportion_have_nickname


def scale_copy_from_household_member(data: pd.DataFrame, column_name: str) -> Any:
    original_column = data[column_name]
    copy_column = data[COPY_HOUSEHOLD_MEMBER_COLS[column_name]]
    original_column_not_missing = (original_column != "") & (original_column.notna())
    eligible = (copy_column != "") & (copy_column.notna()) & original_column_not_missing
    proportion_eligible: float = eligible.sum() / original_column_not_missing.sum()
    if proportion_eligible == 0.0:
        return 0.0
    return 1 / proportion_eligible


####################
# Helper functions #
####################


@cache
def load_nicknames_data() -> pd.DataFrame:
    # Load and format nicknames dataset
    nicknames = pd.read_csv(paths.NICKNAMES_DATA)
    nicknames = nicknames.apply(lambda x: x.astype(str).str.title()).set_index("name")
    nicknames = nicknames.replace("Nan", np.nan)
    return nicknames


def get_options_for_column(column_name: str) -> pd.Series[Any]:
    """
    For a column that has a set list of options, returns that set of options as
    a Series.
    Should only be passed a column that has options (i.e. should not be a free-form
    string column such as first name, or a numeric column), or it will error.
    """
    from pseudopeople.schema_entities import COLUMNS

    selection_type = {
        COLUMNS.employer_state.name: COLUMNS.state.name,
        COLUMNS.mailing_state.name: COLUMNS.state.name,
    }.get(column_name, column_name)

    selection_options = load_incorrect_select_options()
    return selection_options.loc[selection_options[selection_type].notna(), selection_type]


@cache
def load_incorrect_select_options() -> pd.DataFrame:
    return pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)

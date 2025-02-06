from typing import NamedTuple

from pseudopeople import (
    column_getters,
    noise_functions,
    noise_scaling,
    output_dtype_getters,
)
from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_level import get_apply_do_not_respond_noise_level


class __NoiseTypes(NamedTuple):
    """Container for all noise types in the order in which they should be applied:
    do_not_respond, omit_row, duplicate_row, leave_blank, choose_wrong_option,
    copy_from_household_member, swap_month_and_day, write_wrong_zipcode_digits,
    misreport_age, write_wrong_digits, use_nickname, use_fake_name,
    make_phonetic_errors, make_ocr_errors, make_typos

    NOTE: Any configuration tree overwrites in these objects are what ends up
    in the "baseline" LayeredConfigTree layer.
    """

    duplicate_with_guardian: RowNoiseType = RowNoiseType(
        "duplicate_with_guardian",
        noise_function=noise_functions.duplicate_with_guardian,
        probability=None,
        additional_parameters={
            Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.02,
            Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0.05,
        },
        get_noise_level=lambda *_: 1.0,
    )
    do_not_respond: RowNoiseType = RowNoiseType(
        "do_not_respond",
        noise_functions.apply_do_not_respond,
        get_noise_level=get_apply_do_not_respond_noise_level,
    )
    omit_row: RowNoiseType = RowNoiseType("omit_row", noise_functions.omit_rows)
    # duplicate_row: RowNoiseType = RowNoiseType("duplicate_row", noise_functions.duplicate_rows)
    leave_blank: ColumnNoiseType = ColumnNoiseType(
        "leave_blank",
        noise_functions.leave_blanks,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_leave_blank,
    )
    choose_wrong_option: ColumnNoiseType = ColumnNoiseType(
        "choose_wrong_option",
        noise_functions.choose_wrong_options,
        noise_level_scaling_function=noise_scaling.scale_choose_wrong_option,
    )
    copy_from_household_member: ColumnNoiseType = ColumnNoiseType(
        "copy_from_household_member",
        noise_functions.copy_from_household_member,
        noise_level_scaling_function=noise_scaling.scale_copy_from_household_member,
        additional_column_getter=column_getters.copy_from_household_member_column_getter,
    )
    swap_month_and_day: ColumnNoiseType = ColumnNoiseType(
        "swap_month_and_day",
        noise_functions.swap_months_and_days,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_always_string,
    )
    write_wrong_zipcode_digits: ColumnNoiseType = ColumnNoiseType(
        "write_wrong_zipcode_digits",
        noise_functions.write_wrong_zipcode_digits,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_always_string,
        additional_parameters={
            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.04, 0.04, 0.20, 0.36, 0.36],
        },
    )
    misreport_age: ColumnNoiseType = ColumnNoiseType(
        "misreport_age",
        noise_functions.misreport_ages,
        additional_parameters={
            Keys.POSSIBLE_AGE_DIFFERENCES: {-2: 0.1, -1: 0.4, 1: 0.4, 2: 0.1}
        },
    )
    write_wrong_digits: ColumnNoiseType = ColumnNoiseType(
        "write_wrong_digits",
        noise_functions.write_wrong_digits,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_always_string,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    use_nickname: ColumnNoiseType = ColumnNoiseType(
        "use_nickname",
        noise_functions.use_nicknames,
        noise_level_scaling_function=noise_scaling.scale_nicknames,
    )
    use_fake_name: ColumnNoiseType = ColumnNoiseType(
        "use_fake_name",
        noise_functions.use_fake_names,
    )
    make_phonetic_errors: ColumnNoiseType = ColumnNoiseType(
        "make_phonetic_errors",
        noise_functions.make_phonetic_errors,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_always_string,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    make_ocr_errors: ColumnNoiseType = ColumnNoiseType(
        "make_ocr_errors",
        noise_functions.make_ocr_errors,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_always_string,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    make_typos: ColumnNoiseType = ColumnNoiseType(
        "make_typos",
        noise_functions.make_typos,
        output_dtype_getter=output_dtype_getters.output_dtype_getter_always_string,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )


NOISE_TYPES: __NoiseTypes = __NoiseTypes()

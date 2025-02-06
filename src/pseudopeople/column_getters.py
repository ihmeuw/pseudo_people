from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS


def copy_from_household_member_column_getter(column_name: str) -> list[str]:
    return [COPY_HOUSEHOLD_MEMBER_COLS[column_name]]

class Keys:
    """Container for all non-dataset standard/repeated key names used in the configuration file"""

    ROW_NOISE: str = "row_noise"  # second layer, eg <dataset>: row_noise: {...}
    COLUMN_NOISE: str = "column_noise"  # second layer, eg <dataset>: column_noise: {...}
    ROW_PROBABILITY: str = "row_probability"
    CELL_PROBABILITY: str = "cell_probability"
    TOKEN_PROBABILITY: str = "token_probability"
    POSSIBLE_AGE_DIFFERENCES: str = "possible_age_differences"
    ZIPCODE_DIGIT_PROBABILITIES: str = "digit_probabilities"
    ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: str = "row_probability_in_households_under_18"
    ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: str = (
        "row_probability_in_college_group_quarters_under_24"
    )


NO_NOISE: str = "no_noise"

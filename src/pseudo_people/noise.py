import pandas as pd
from vivarium import ConfigTree

from pseudo_people.entities import NOISE_TYPES, Form
from pseudo_people.entity_types import ColumnNoiseType, RowNoiseType


def noise_form(
    form: Form,
    form_data: pd.DataFrame,
    configuration: ConfigTree,
) -> pd.DataFrame:
    """
    Adds noise to the input form data. Noise functions are executed in the order
    defined by :py:const: `.NOISE_TYPES`. Row noise functions are applied to the
    whole DataFrame. Column noise functions will be applied to each column that
    is pertinent to it.

    Noise levels are determined by the noise_config.
    :param form:
        Form needing to be noised
    :param form_data:
        Clean data input which needs to be noised.
    :param configuration:
        Object to configure noise levels
    :return:
        Noised form data
    """

    for noise_type in NOISE_TYPES:
        noise_configuration = configuration[form][noise_type]
        if isinstance(noise_type, RowNoiseType):
            # Apply row noise
            form_data = noise_type(form_data, noise_configuration)

        elif isinstance(noise_type, ColumnNoiseType):
            # Apply column noise to each column as appropriate
            for column in form_data.columns:
                if column not in noise_configuration:
                    continue

                column_configuration = noise_configuration[column]
                form_data[column] = noise_type(form_data[column], column_configuration)
        else:
            raise TypeError(
                f"Invalid noise type. Allowed types are {RowNoiseType} and "
                f"{ColumnNoiseType}. Provided {type(noise_type)}."
            )

    return form_data

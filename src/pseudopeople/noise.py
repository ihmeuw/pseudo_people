import pandas as pd
from vivarium import ConfigTree

from pseudopeople.entities import NOISE_TYPES, Form
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.utilities import get_randomness_stream


def noise_form(
    form: Form,
    form_data: pd.DataFrame,
    configuration: ConfigTree,
    seed: int,
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
    :param seed:
        Seed for controlling randomness
    :return:
        Noised form data
    """
    randomness = get_randomness_stream(form, seed)

    noise_configuration = configuration[form.value]
    for noise_type in NOISE_TYPES:
        if isinstance(noise_type, RowNoiseType):
            # Apply row noise
            form_data = noise_type(form_data, noise_configuration, randomness)

        elif isinstance(noise_type, ColumnNoiseType):
            columns_to_noise = [
                col
                for col in noise_configuration
                if col in form_data.columns
                and noise_type.name in noise_configuration[col]
            ]
            # Apply column noise to each column as appropriate
            for column in columns_to_noise:
                form_data[column] = noise_type(
                    form_data[column],
                    noise_configuration[column][noise_type.name],
                    randomness,
                    column,
                )
        else:
            raise TypeError(
                f"Invalid noise type. Allowed types are {RowNoiseType} and "
                f"{ColumnNoiseType}. Provided {type(noise_type)}."
            )

    return form_data

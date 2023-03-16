import sys
from pathlib import Path
from typing import Union

import pandas as pd

from pseudopeople.entities import Form
from pseudopeople.noise import noise_form
from pseudopeople.utilities import get_default_configuration


# TODO: add year as parameter to select the year of the decennial census to generate
# TODO: add default path: have the package install the small data in a known location and then to make this
#  parameter optional, with the default being the location of the small data that is installed with the package
def generate_decennial_census(path: Union[Path, str], seed: int = 0):
    """
    Generates a noised decennial census data from un-noised data.

    :param path: A path to the un-noised source census data
    :param seed: An integer seed for randomness
    :return: A pd.DataFrame of noised census data
    """
    configuration = get_default_configuration()
    data = pd.read_csv(path)
    return noise_form(Form.CENSUS, data, configuration, seed)


# Manual testing helper
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        my_path = Path(args[0])
        src = pd.read_csv(my_path)
        out = generate_decennial_census(my_path)
        diff = src[
            ~src.astype(str).apply(tuple, 1).isin(out.astype(str).apply(tuple, 1))
        ]  # get all changed rows
        print(out.head())
        print(diff)

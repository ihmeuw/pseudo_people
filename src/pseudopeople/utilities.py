from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from vivarium.framework.configuration import ConfigTree
from vivarium.framework.randomness import RandomnessStream, random

from pseudopeople.schema_entities import Form


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)


def get_configuration(user_yaml_path: Union[Path, str] = None) -> ConfigTree:
    """
    Gets a noising configuration ConfigTree, optionally overridden by a user-provided YAML.

    :param user_yaml_path: A path to the YAML file defining user overrides for the defaults
    :return: a ConfigTree object of the noising configuration
    """
    import pseudopeople

    default_config_layers = [
        "base",
        "user",
    ]
    noising_configuration = ConfigTree(
        data=Path(pseudopeople.__file__).resolve().parent / "default_configuration.yaml",
        layers=default_config_layers,
    )
    if user_yaml_path:
        noising_configuration.update(user_yaml_path, layer="user")
    return noising_configuration


def vectorized_choice(
    options: Union[list, pd.Series],
    n_to_choose: int,
    randomness_stream: RandomnessStream = None,
    weights: Union[list, pd.Series] = None,
    additional_key: Any = None,
    random_seed: int = None,
):
    """
    Function that takes a list of options and uses Vivarium common random numbers framework to make a given number
    of razndom choice selections.

    :param options: List and series of possible values to choose
    :param n_to_choose: Number of choices to make, the length of the returned array of values
    :param randomness_stream: RandomnessStream being used for Vivarium's CRN framework
    :param weights: List or series containing weights for each options
    :param additional_key: Key to pass to randomness_stream
    :param random_seed: Seed to pass to randomness_stream.
    Note additional_key and random_seed are used to make calls using a RandomnessStream unique

    returns: ndarray
    """
    if not randomness_stream and (additional_key == None and random_seed == None):
        raise RuntimeError(
            "An additional_key and a random_seed are required in 'vectorized_choice'"
            + "if no RandomnessStream is passed in"
        )
    if weights is None:
        n = len(options)
        weights = np.ones(n) / n
    if isinstance(weights, list):
        weights = np.array(weights)
    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pd.Index(np.arange(n_to_choose))
    if randomness_stream is None:
        # Generate an additional_key on-the-fly and use that in randomness.random
        additional_key = f"{additional_key}_{random_seed}"
        probs = random(str(additional_key), index)
    else:
        probs = randomness_stream.get_draw(index, additional_key=additional_key)

    # build cdf based on weights
    pmf = weights / sum(weights)
    cdf = np.cumsum(pmf)

    # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
    chosen_indices = np.searchsorted(cdf, probs, side="right")
    return np.take(options, chosen_indices)


def get_index_to_noise(
    column: pd.Series,
    noise_level: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Index:
    """
    Function that takes a series and returns a pd.Index that chosen by Vivarium Common Random Number to be noised.
    """

    # Get rows to noise
    not_empty_idx = column.index[(column != "") & (column.notna())]
    to_noise_idx = randomness_stream.filter_for_probability(
        not_empty_idx,
        probability=noise_level,
        additional_key=additional_key,
    )

    return to_noise_idx

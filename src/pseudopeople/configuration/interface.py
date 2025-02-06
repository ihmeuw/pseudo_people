from __future__ import annotations

from pathlib import Path
from typing import Any

from pseudopeople.configuration.generator import get_configuration


def get_config(overrides: Path | str | dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Function that returns the pseudopeople configuration containing all
    default values. To get the default probability of nonresponse in the
    Decennial Census dataset:

    .. code-block:: pycon

        >>> import pseudopeople as psp
        >>> psp.get_config()['decennial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.0145}

    To view that same part of the configuration after applying a user
    override:

    .. code-block:: pycon

        >>> overrides = {'decennial_census': {'row_noise': {'do_not_respond': {'row_probability': 0.1}}}}
        >>> psp.get_config(overrides)['decenial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.1}

    :param overrides:

        An optional set of overrides to the default configuration. Can
        be a (nested) Python dictionary mapping noise type parameters to
        the desired override values, a path to a YAML file with the same
        nested structure (see the :ref:`configuration structure
        <configuration_structure>` section of the documentation for
        details), or the special sentinel value `pseudopeople.NO_NOISE`,
        which will return a configuration in which all configurable
        noise is set to zero. When passing a dictionary or YAML file, it
        is not necessary to provide a complete configuration; any
        configuration parameters not specified in `overrides` will be
        filled in with the default values.

    :return:

        A complete configuration dictionary.

    :raises ConfigurationError:

        An invalid configuration is passed with `overrides`.

    """
    config: dict[str, Any] = get_configuration(overrides).to_dict()
    return config

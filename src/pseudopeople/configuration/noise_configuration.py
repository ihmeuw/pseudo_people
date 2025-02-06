from __future__ import annotations

from typing import Any

from layered_config_tree import LayeredConfigTree
from layered_config_tree.types import InputData

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, NoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS

COLUMN_NOISE_TYPES = [
    noise_type.name for noise_type in NOISE_TYPES if isinstance(noise_type, ColumnNoiseType)
]
ROW_NOISE_TYPES = [
    noise_type.name for noise_type in NOISE_TYPES if isinstance(noise_type, RowNoiseType)
]


class NoiseConfiguration:
    def __init__(self, config: LayeredConfigTree):
        self._config = config

    def to_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = self._config.to_dict()
        return config_dict

    def get_value(
        self,
        dataset: str,
        noise_type: str,
        parameter_name: str,
        column_name: str | None = None,
    ) -> float | int | list[float] | dict[int, float]:
        config = self._config
        try:
            dataset_config: LayeredConfigTree = config.get_tree(dataset)
        except:
            raise ValueError(
                f"{dataset} was not found in configuration. "
                f"Available datasets are {list(config.keys())}"
            )

        # row noise
        if noise_type in ROW_NOISE_TYPES:
            if column_name:
                raise ValueError(
                    f"You cannot provide both a row noise type ({noise_type}) and a column name ({column_name}) simultaneously."
                )
            config = dataset_config.get_tree("row_noise")
        # column noise
        elif noise_type in COLUMN_NOISE_TYPES:
            if not column_name:
                raise ValueError(
                    f"You must provide a column name when using a column noise type ({noise_type} in your case)."
                )
            all_column_configs: LayeredConfigTree = dataset_config.get_tree("column_noise")
            if column_name not in all_column_configs:
                raise ValueError(
                    f"The column name {column_name} was not found in your config. "
                    f"Available columns are {list(all_column_configs.keys())}."
                )
            config = all_column_configs.get_tree(column_name)
        # unknown noise type
        else:
            raise ValueError(
                f"Your noise type {noise_type} was not found in row noise types or column noise types. "
                f"Available row noise types are {ROW_NOISE_TYPES}. "
                f"Available column noise types are {COLUMN_NOISE_TYPES}."
            )
        # get value
        parameter_tree: LayeredConfigTree = config.get_tree(noise_type)
        if parameter_name not in parameter_tree:
            raise ValueError(
                f"The parameter {parameter_name} was not found for {noise_type} in the configuration. "
                f"Available parameters are {list(parameter_tree.keys())}."
            )
        noise_value: int | float | LayeredConfigTree = parameter_tree.get(parameter_name)
        if isinstance(noise_value, LayeredConfigTree):
            # TODO: [MIC-5500] store dicts in LayeredConfigTree without converting to LayeredConfigTree
            converted_noise_value: dict[int, float] = noise_value.to_dict()  # type: ignore [assignment]
            return converted_noise_value
        else:
            return noise_value

    def get_row_probability(self, dataset: str, noise_type: str) -> int | float:
        value = self.get_value(dataset, noise_type, parameter_name="row_probability")
        if not isinstance(value, int) and not isinstance(value, float):
            raise ValueError(
                f"Row probabilities are expected to contain ints or floats. Your config returned {type(value)}."
            )
        return value

    def get_cell_probability(
        self, dataset: str, noise_type: str, column_name: str
    ) -> int | float:
        value = self.get_value(
            dataset, noise_type, parameter_name="cell_probability", column_name=column_name
        )
        if not isinstance(value, int) and not isinstance(value, float):
            raise ValueError(
                f"Cell probabilities are expected to contain ints or floats. Your config returned {type(value)}."
            )
        return value

    def get_token_probability(
        self, dataset: str, noise_type: str, column_name: str
    ) -> int | float:
        value = self.get_value(
            dataset, noise_type, parameter_name="token_probability", column_name=column_name
        )
        if not isinstance(value, int) and not isinstance(value, float):
            raise ValueError(
                f"Token probabilities are expected to contain ints or floats. Your config returned {type(value)}."
            )
        return value

    def get_zipcode_digit_probabilities(self, dataset: str, column_name: str) -> list[float]:
        values = self.get_value(
            dataset,
            "write_wrong_zipcode_digits",
            parameter_name="digit_probabilities",
            column_name=column_name,
        )
        if not isinstance(values, list) or not all(
            isinstance(value, float) for value in values
        ):
            raise ValueError(
                f"Zipcode digit probabilities are expected to be a list of floats. Your config returned {type(values)}."
            )
        return values

    def get_duplicate_with_guardian_probabilities(
        self, dataset: str, parameter_name: str
    ) -> int | float:
        if (
            parameter_name != "row_probability_in_households_under_18"
            and parameter_name != "row_probability_in_college_group_quarters_under_24"
        ):
            raise ValueError(
                f"Parameter name must be 'row_probability_in_households_under_18' or 'row_probability_in_college_group_quarters_under_24' when getting duplicate with guardian probabilities. You provided {parameter_name}."
            )
        value = self.get_value(dataset, "duplicate_with_guardian", parameter_name)
        if not isinstance(value, int) and not isinstance(value, float):
            raise ValueError(
                f"Duplicate with guardian probabilities are expected to be ints or floats. Your config returned {type(value)}."
            )
        return value

    def get_misreport_ages_probabilities(
        self, dataset: str, column_name: str
    ) -> dict[int, float]:
        value = self.get_value(
            dataset, "misreport_age", Keys.POSSIBLE_AGE_DIFFERENCES, column_name
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"Misreport age probabilities are expected to be a dict. Your config returned {type(value)}."
            )
        return value

    def has_noise_type(
        self, dataset_name: str, noise_type: str, column: str | None = None
    ) -> bool:
        dataset_config = self.to_dict()[dataset_name]
        if column is not None:
            has_noise_type = noise_type in dataset_config.get("column_noise", {}).get(
                column, {}
            )
        else:
            has_noise_type = noise_type in dataset_config.get("row_noise", {})
        return has_noise_type

    def has_parameter(
        self,
        dataset: str,
        noise_type: str,
        parameter_name: str,
        column_name: str | None = None,
    ) -> bool:
        if column_name:
            has_parameter = parameter_name in self.to_dict().get(dataset, {}).get(
                "column_noise", {}
            ).get(column_name, {}).get(noise_type, {})
        else:
            has_parameter = parameter_name in self.to_dict().get(dataset, {}).get(
                "row_noise", {}
            ).get(noise_type, {})

        return has_parameter

    def _update(self, data: InputData) -> None:
        self._config.update(data)

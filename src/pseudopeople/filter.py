from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataFilter:
    column_name: str
    operator: str
    value: str | int | pd.Timestamp

    def to_tuple(self) -> tuple[str, str, str | int | pd.Timestamp]:
        return self.column_name, self.operator, self.value

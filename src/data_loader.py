import os
import pandas as pd
from exceptions import DataLoadError, SchemaMismatchError


class CSVDataLoader:
    def __init__(self, filepath, required_cols):
        self.filepath = filepath
        self.required_cols = required_cols
        self.data = None

    def load(self):
        if not os.path.exists(self.filepath):
            raise DataLoadError(f"File not found: {self.filepath}")
        try:
            df = pd.read_csv(self.filepath)
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Couldn't parse {self.filepath}: {e}") from e

        self._validate(df)
        # sort by x so downstream code can assume ordered data
        self.data = df.sort_values("x").reset_index(drop=True)
        return self.data

    def _validate(self, df):
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise SchemaMismatchError(f"{self.filepath} is missing columns: {missing}")


class TrainingDataLoader(CSVDataLoader):
    def __init__(self, filepath):
        super().__init__(filepath, ["x", "y1", "y2", "y3", "y4"])

    def _validate(self, df):
        super()._validate(df)
        if df.empty:
            raise SchemaMismatchError(f"{self.filepath} has no rows")


class IdealDataLoader(CSVDataLoader):
    def __init__(self, filepath):
        cols = ["x"] + [f"y{i}" for i in range(1, 51)]
        super().__init__(filepath, cols)

    def _validate(self, df):
        super()._validate(df)
        # 50 ideal functions + x column = 51
        if df.shape[1] != 51:
            raise SchemaMismatchError(
                f"Expected 51 columns (x + y1..y50), got {df.shape[1]}"
            )


class TestDataLoader(CSVDataLoader):
    def __init__(self, filepath):
        super().__init__(filepath, ["x", "y"])

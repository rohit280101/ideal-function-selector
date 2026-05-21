import os
import pandas as pd
from sqlalchemy import create_engine
from exceptions import DataLoadError


class DatabaseManager:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)

    def write(self, df, table_name, if_exists="replace"):
        try:
            df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
        except Exception as e:
            raise DataLoadError(f"Failed writing '{table_name}': {e}") from e

    def read(self, table_name):
        return pd.read_sql_table(table_name, con=self.engine)

    def save_training(self, df):
        self.write(df, "training_data")

    def save_ideal(self, df):
        self.write(df, "ideal_functions")

    def save_mapping(self, df):
        self.write(df, "test_mapping")

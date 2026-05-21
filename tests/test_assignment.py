import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import IdealDataLoader, TestDataLoader, TrainingDataLoader
from database import DatabaseManager
from exceptions import DataLoadError, SchemaMismatchError
from function_selector import IdealFunctionSelector, TestMapper


def tmp_csv(df):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


class TestLoaders(unittest.TestCase):

    def test_training_loads_ok(self):
        df = pd.DataFrame({"x":[1,2],"y1":[0,0],"y2":[0,0],"y3":[0,0],"y4":[0,0]})
        p = tmp_csv(df)
        try:
            result = TrainingDataLoader(p).load()
            self.assertEqual(result.shape, (2, 5))
        finally:
            os.unlink(p)

    def test_test_loads_ok(self):
        df = pd.DataFrame({"x":[0,1],"y":[3,4]})
        p = tmp_csv(df)
        try:
            self.assertEqual(TestDataLoader(p).load().shape, (2, 2))
        finally:
            os.unlink(p)

    def test_missing_file_raises(self):
        with self.assertRaises(DataLoadError):
            TrainingDataLoader("/no/such/file.csv").load()

    def test_wrong_schema_raises(self):
        df = pd.DataFrame({"x":[1],"y1":[0]})  # missing y2-y4
        p = tmp_csv(df)
        try:
            with self.assertRaises(SchemaMismatchError):
                TrainingDataLoader(p).load()
        finally:
            os.unlink(p)

    def test_ideal_needs_51_cols(self):
        data = {"x":[1,2]}
        for i in range(1, 51):
            data[f"y{i}"] = [float(i), float(i)]
        p = tmp_csv(pd.DataFrame(data))
        try:
            result = IdealDataLoader(p).load()
            self.assertEqual(result.shape[1], 51)
        finally:
            os.unlink(p)


class TestSelector(unittest.TestCase):

    def test_picks_exact_match(self):
        x = np.linspace(0, 5, 11)
        train = pd.DataFrame({"x":x,"y1":x*2,"y2":x+1,"y3":x**2,"y4":np.sin(x)})

        ideal_data = {"x": x}
        for i in range(1, 51):
            ideal_data[f"y{i}"] = x * (i + 10)
        ideal_data["y1"] = x * 2
        ideal_data["y2"] = x + 1
        ideal_data["y3"] = x ** 2
        ideal_data["y4"] = np.sin(x)
        ideal = pd.DataFrame(ideal_data)

        sel = IdealFunctionSelector(train, ideal)
        fits = sel.select()

        self.assertEqual(fits["y1"], "y1")
        self.assertEqual(fits["y2"], "y2")
        for v in sel.max_deviations.values():
            self.assertAlmostEqual(v, 0.0, places=6)


class TestMappingLogic(unittest.TestCase):

    def _setup(self):
        x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        ideal = pd.DataFrame({"x":x,"y10":x*2,"y20":x+1,"y30":x**2,"y40":-x})
        fits  = {"y1":"y10","y2":"y20","y3":"y30","y4":"y40"}
        maxdev = {"y1":1.0,"y2":1.0,"y3":1.0,"y4":1.0}
        return ideal, fits, maxdev

    def test_within_threshold_matched(self):
        ideal, fits, maxdev = self._setup()
        # at x=2, y10=4; y=4.5 → delta=0.5 < 1*sqrt(2)
        df = pd.DataFrame({"x":[2.0],"y":[4.5]})
        result = TestMapper(df, ideal, fits, maxdev).map_points()
        self.assertEqual(result.iloc[0]["ideal_func"], "y10")
        self.assertAlmostEqual(result.iloc[0]["delta_y"], 0.5)

    def test_outside_threshold_unmatched(self):
        ideal, fits, maxdev = self._setup()
        df = pd.DataFrame({"x":[2.0],"y":[99.0]})
        result = TestMapper(df, ideal, fits, maxdev).map_points()
        self.assertEqual(result.iloc[0]["ideal_func"], "")
        self.assertTrue(math.isnan(result.iloc[0]["delta_y"]))

    def test_exact_boundary_accepted(self):
        ideal, fits, maxdev = self._setup()
        df = pd.DataFrame({"x":[2.0],"y":[4.0 + math.sqrt(2)]})
        result = TestMapper(df, ideal, fits, maxdev).map_points()
        self.assertEqual(result.iloc[0]["ideal_func"], "y10")


class TestDatabase(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(os.path.join(tmp, "test.db"))
            df = pd.DataFrame({"x":[1.0,2.0],"y":[3.0,4.0]})
            db.write(df, "demo")
            back = db.read("demo")
            self.assertEqual(back.shape, df.shape)
            self.assertListEqual(list(back.columns), list(df.columns))


if __name__ == "__main__":
    unittest.main()

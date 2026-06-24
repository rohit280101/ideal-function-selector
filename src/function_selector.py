import math
import pandas as pd
from exceptions import NoIdealFunctionError

# threshold multiplier from the assignment spec
SQRT2 = math.sqrt(2)


class IdealFunctionSelector:
    def __init__(self, training_df, ideal_df):
        self.training_df = training_df
        self.ideal_df = ideal_df
        self.best_fits = {}
        self.max_deviations = {}

    def select(self):
        # prefix columns before merging so t_y1 and i_y1 don't collide
        train = self.training_df.rename(
            columns={c: f"t_{c}" for c in self.training_df.columns if c != "x"}
        )
        ideal = self.ideal_df.rename(
            columns={c: f"i_{c}" for c in self.ideal_df.columns if c != "x"}
        )
        merged = pd.merge(train, ideal, on="x")

        for tcol in ["y1", "y2", "y3", "y4"]:
            train_vals = merged[f"t_{tcol}"]
            best_col, best_sse, best_max = None, math.inf, math.inf

            for n in range(1, 51):
                icol = f"y{n}"
                diff = train_vals - merged[f"i_{icol}"]
                sse = float((diff ** 2).sum())
                if sse < best_sse:
                    best_sse  = sse
                    best_col  = icol
                    best_max  = float(diff.abs().max())

            if best_col is None:
                raise NoIdealFunctionError(f"No ideal function found for {tcol}")

            self.best_fits[tcol] = best_col
            self.max_deviations[tcol] = best_max

        return self.best_fits


class TestMapper:
    def __init__(self, test_df, ideal_df, best_fits, max_deviations):
        self.test_df = test_df
        self.ideal_df = ideal_df.set_index("x")
        self.best_fits = best_fits
        self.max_deviations = max_deviations

    def map_points(self):
        rows = []
        for _, pt in self.test_df.iterrows():
            x, y = float(pt["x"]), float(pt["y"])

            if x not in self.ideal_df.index:
                rows.append({"x": x, "y": y, "delta_y": float("nan"), "ideal_func": ""})
                continue

            best_func, best_delta = "", math.inf
            for tcol, icol in self.best_fits.items():
                delta = abs(y - float(self.ideal_df.loc[x, icol]))
                threshold = self.max_deviations[tcol] * SQRT2
                if delta <= threshold and delta < best_delta:
                    best_delta = delta
                    best_func  = icol

            rows.append({
                "x": x,
                "y": y,
                "delta_y": best_delta if best_func else float("nan"),
                "ideal_func": best_func,
            })

        return pd.DataFrame(rows, columns=["x", "y", "delta_y", "ideal_func"])

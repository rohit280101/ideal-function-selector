import os
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.plotting import figure

COLORS = {"y1": "#1f77b4", "y2": "#ff7f0e", "y3": "#2ca02c", "y4": "#d62728"}


class Visualizer:
    def __init__(self, training_df, ideal_df, mapping_df, best_fits, output_dir="output"):
        self.training = training_df
        self.ideal = ideal_df
        self.mapping = mapping_df
        self.best_fits = best_fits
        self.out = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_vs_ideal(self):
        plots = []
        for tcol, icol in self.best_fits.items():
            p = figure(
                title=f"{tcol}  →  {icol}",
                x_axis_label="x", y_axis_label="y",
                width=440, height=330,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            p.scatter(
                self.training["x"], self.training[tcol],
                size=4, color=COLORS[tcol], alpha=0.7,
                legend_label=f"train {tcol}",
            )
            p.line(
                self.ideal["x"], self.ideal[icol],
                line_width=2, color="#222",
                legend_label=f"ideal {icol}",
            )
            p.legend.location = "top_left"
            p.legend.background_fill_alpha = 0.7
            plots.append(p)

        grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]], merge_tools=False)
        path = os.path.join(self.out, "training_vs_ideal.html")
        output_file(path, title="Training vs Ideal Functions")
        save(grid)
        return path

    def plot_test_mapping(self):
        p = figure(
            title="Test points mapped to ideal functions",
            x_axis_label="x", y_axis_label="y",
            width=900, height=560,
            tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        )

        # background ideal function lines
        for tcol, icol in self.best_fits.items():
            p.line(
                self.ideal["x"], self.ideal[icol],
                line_width=1.5, color=COLORS[tcol], alpha=0.5,
                legend_label=f"{icol} (←{tcol})",
            )

        # matched points
        matched = self.mapping[self.mapping["ideal_func"] != ""].copy()
        if not matched.empty:
            color_map = {v: COLORS[k] for k, v in self.best_fits.items()}
            matched["color"] = matched["ideal_func"].map(color_map).fillna("grey")
            max_dev = matched["delta_y"].max() or 1.0
            matched["pt_size"] = 6 + (matched["delta_y"] / max_dev) * 10
            from bokeh.models import ColumnDataSource
            p.scatter(
                "x", "y", source=ColumnDataSource(matched),
                size="pt_size", color="color", alpha=0.85,
                legend_label="matched",
            )
            p.add_tools(HoverTool(tooltips=[
                ("x", "@x{0.00}"), ("y", "@y{0.00}"),
                ("Δy", "@delta_y{0.000}"), ("func", "@ideal_func"),
            ]))

        # unmatched points
        unmatched = self.mapping[self.mapping["ideal_func"] == ""]
        if not unmatched.empty:
            p.scatter(
                unmatched["x"], unmatched["y"],
                size=6, color="grey", marker="x", alpha=0.5,
                legend_label="unmatched",
            )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        path = os.path.join(self.out, "test_mapping.html")
        output_file(path, title="Test Mapping")
        save(p)
        return path

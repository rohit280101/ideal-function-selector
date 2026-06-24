import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import TrainingDataLoader, IdealDataLoader, TestDataLoader
from database import DatabaseManager
from function_selector import IdealFunctionSelector, TestMapper
from visualizer import Visualizer


def run(
    train_path="data/train.csv",
    ideal_path="data/ideal.csv",
    test_path="data/test.csv",
    db_path="output/assignment.db",
    output_dir="output",
):
    print("=" * 55)
    print("  Programming with Python - Written Assignment")
    print("=" * 55)

    print("\nLoading datasets...")
    train_df = TrainingDataLoader(train_path).load()
    ideal_df = IdealDataLoader(ideal_path).load()
    test_df  = TestDataLoader(test_path).load()
    print(f"  train {train_df.shape}, ideal {ideal_df.shape}, test {test_df.shape}")

    print("\nSaving raw tables to database...")
    db = DatabaseManager(db_path)
    db.save_training(train_df)
    db.save_ideal(ideal_df)
    print(f"  -> {db_path}")

    print("\nSelecting best-fit ideal functions (least squares)...")
    selector = IdealFunctionSelector(train_df, ideal_df)
    best_fits = selector.select()
    for tcol, icol in best_fits.items():
        print(f"  {tcol} -> {icol}  (max deviation = {selector.max_deviations[tcol]:.4f})")

    print("\nMapping test points...")
    mapper  = TestMapper(test_df, ideal_df, best_fits, selector.max_deviations)
    mapping = mapper.map_points()
    n_matched = (mapping["ideal_func"] != "").sum()
    print(f"  {n_matched}/{len(mapping)} points matched")

    db.save_mapping(mapping)

    print("\nGenerating plots...")
    viz = Visualizer(train_df, ideal_df, mapping, best_fits, output_dir)
    print(f"  {viz.plot_training_vs_ideal()}")
    print(f"  {viz.plot_test_mapping()}")

    print("\nAll done.\n")


if __name__ == "__main__":
    run()

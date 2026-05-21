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
    print("  Programming with Python — Written Assignment")
    print("=" * 55)

    # Load data
    print("\n[1/6] Loading datasets...")
    train_df = TrainingDataLoader(train_path).load()
    ideal_df = IdealDataLoader(ideal_path).load()
    test_df  = TestDataLoader(test_path).load()
    print(f"      train {train_df.shape}  ideal {ideal_df.shape}  test {test_df.shape}")

    # Persist source tables
    print("\n[2/6] Writing to database...")
    db = DatabaseManager(db_path)
    db.save_training(train_df)
    db.save_ideal(ideal_df)
    print(f"      Saved → {db_path}")

    # Select best-fit ideal functions
    print("\n[3/6] Running least-squares selection...")
    selector = IdealFunctionSelector(train_df, ideal_df)
    best_fits = selector.select()
    for tcol, icol in best_fits.items():
        print(f"      {tcol} → {icol}  (max|dev| = {selector.max_deviations[tcol]:.4f})")

    # Map test points
    print("\n[4/6] Mapping test points...")
    mapper  = TestMapper(test_df, ideal_df, best_fits, selector.max_deviations)
    mapping = mapper.map_points()
    matched = (mapping["ideal_func"] != "").sum()
    print(f"      {matched}/{len(mapping)} points matched")

    # Persist mapping
    print("\n[5/6] Saving test mapping...")
    db.save_mapping(mapping)

    # Visualise
    print("\n[6/6] Generating visualisations...")
    viz = Visualizer(train_df, ideal_df, mapping, best_fits, output_dir)
    print(f"      {viz.plot_training_vs_ideal()}")
    print(f"      {viz.plot_test_mapping()}")

    print("\nDone.\n")


if __name__ == "__main__":
    run()

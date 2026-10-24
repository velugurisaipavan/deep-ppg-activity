"""Saved-model inference: HR and activity labels are never read."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .activity import probabilities
from .model import ActivityAwareCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--subject-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config = json.loads((args.run_dir / "config.json").read_text())
    with np.load(args.subject_cache, allow_pickle=False) as archive:
        features, spectra, starts = archive["features"], archive["spectra"], archive["start_seconds"]
    h = config["history"]
    rows = np.arange(h - 1, len(features))
    for lag in range(1, h):
        rows = rows[starts[rows] - starts[rows - lag] == 2 * lag]
    if not len(rows):
        raise ValueError("No complete histories available")
    if config["model"] == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(args.run_dir / "xgboost.json")
        result = model.predict(features[rows])
    else:
        model = ActivityAwareCNN(h, config["activity_dim"])
        model.load_state_dict(torch.load(args.run_dir / "cnn.pt", map_location="cpu", weights_only=True))
        model.eval()
        activity = np.zeros((len(features), 9), dtype=np.float32)
        if config["activity_dim"]:
            # joblib artifacts must be trusted; loading one can execute code.
            rf = joblib.load(args.run_dir / "activity.joblib")
            activity = probabilities(rf, features)
        batches = []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                selection = rows[start:start + args.batch_size]
                x = np.stack([spectra[i-h+1:i+1].transpose(1, 0, 2) for i in selection])
                pred = model(torch.from_numpy(x).float(), torch.from_numpy(activity[selection])).numpy()
                batches.append(pred * config["target_std"] + config["target_mean"])
        result = np.concatenate(batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_start_seconds", "predicted_bpm"])
        writer.writerows(zip(starts[rows], result))


if __name__ == "__main__":
    main()

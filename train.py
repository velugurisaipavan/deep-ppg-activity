"""Train a single subject-disjoint fold, with validation-only model selection."""
import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from .activity import fit_activity, probabilities
from .data import WindowDataset, eligible_rows, load_cache
from .model import ActivityAwareCNN
from .preprocess import ACTIVITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-subject", default="S15")
    parser.add_argument("--val-subject", default="S14")
    parser.add_argument("--model", choices=["cnn", "xgboost"], default="cnn")
    parser.add_argument("--no-activity", action="store_true")
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.history, args.epochs, args.batch_size, args.patience, args.rf_trees) < 1 or args.learning_rate <= 0:
        parser.error("Numeric hyperparameters must be positive")
    if args.test_subject == args.val_subject:
        parser.error("Validation and test subjects must differ")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Choose an empty output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = load_cache(args.cache_dir)
    eligible = eligible_rows(data, args.history)
    train = eligible[~np.isin(data["subject"][eligible], [args.test_subject, args.val_subject])]
    val = eligible[data["subject"][eligible] == args.val_subject]
    test = eligible[data["subject"][eligible] == args.test_subject]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Empty split; check subjects and history length")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(train_subjects=np.unique(data["subject"][train]).tolist(),
                  versions=dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__),
                  protocol="subject-disjoint reconstruction; not original paper split")
    activity_prob = np.zeros((len(data["hr"]), 9), dtype=np.float32)
    activity_accuracy = None
    if args.model == "cnn" and not args.no_activity:
        rf, oof = fit_activity(data["features"][train], data["activity"][train], data["subject"][train], args.seed, args.rf_trees)
        activity_prob[train] = oof
        activity_prob[val] = probabilities(rf, data["features"][val])
        activity_prob[test] = probabilities(rf, data["features"][test])
        activity_accuracy = float(accuracy_score(data["activity"][test], activity_prob[test].argmax(axis=1)))
        joblib.dump(rf, args.out_dir / "activity.joblib")
    history_log = []
    if args.model == "xgboost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=.03, tree_method="hist",
                                 objective="reg:squarederror", eval_metric="mae", early_stopping_rounds=args.patience,
                                 random_state=args.seed, n_jobs=4)
        estimator.fit(data["features"][train], data["hr"][train],
                      eval_set=[(data["features"][val], data["hr"][val])], verbose=False)
        prediction = estimator.predict(data["features"][test])
        estimator.save_model(args.out_dir / "xgboost.json")
        config["best_iteration"] = int(estimator.best_iteration)
    else:
        device = torch.device(args.device)
        mean = float(data["hr"][train].mean())
        std = max(float(data["hr"][train].std()), 1e-6)
        config.update(target_mean=mean, target_std=std, activity_dim=0 if args.no_activity else 9)
        model = ActivityAwareCNN(args.history, config["activity_dim"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.MSELoss()

        def loader(rows, shuffle=False):
            return DataLoader(WindowDataset(data, rows, activity_prob, args.history, mean, std),
                              batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

        def infer(rows):
            model.eval()
            result = []
            with torch.no_grad():
                for x, activity, _ in loader(rows):
                    result.append(model(x.to(device), activity.to(device)).cpu().numpy() * std + mean)
            return np.concatenate(result)

        best, stale = float("inf"), 0
        train_loader = loader(train, True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.
            for x, activity, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x.to(device), activity.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(x)
            score = float(mean_absolute_error(data["hr"][val], infer(val)))
            history_log.append(dict(epoch=epoch, train_normalized_mse=total / len(train), validation_mae_bpm=score))
            print(f"Epoch {epoch}: validation MAE {score:.4f} BPM")
            if score < best:
                best, stale = score, 0
                config["best_epoch"] = epoch
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.out_dir / "cnn.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(torch.load(args.out_dir / "cnn.pt", weights_only=True, map_location=device))
        prediction = infer(test)
    metrics = dict(n_test_windows=len(test), test_subject=args.test_subject,
                   mae_bpm=float(mean_absolute_error(data["hr"][test], prediction)),
                   rmse_bpm=float(np.sqrt(mean_squared_error(data["hr"][test], prediction))),
                   activity_accuracy=activity_accuracy, per_activity={})
    for code in np.unique(data["activity"][test]):
        mask = data["activity"][test] == code
        metrics["per_activity"][str(code)] = dict(name=ACTIVITIES[int(code)], n=int(mask.sum()),
            mae_bpm=float(mean_absolute_error(data["hr"][test][mask], prediction[mask])))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    if history_log:
        with (args.out_dir / "training.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history_log[0]))
            writer.writeheader()
            writer.writerows(history_log)
    with (args.out_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "window_start_seconds", "true_bpm", "predicted_bpm", "true_activity", "predicted_activity"])
        for row, pred in zip(test, prediction):
            activity = int(activity_prob[row].argmax()) if activity_accuracy is not None else ""
            writer.writerow([data["subject"][row], int(data["start_seconds"][row]), float(data["hr"][row]),
                             float(pred), int(data["activity"][row]), activity])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

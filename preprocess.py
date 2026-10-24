"""Read official trusted PPG-DaLiA pickles; build aligned wrist-only features."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ACTIVITIES = {0: "transition", 1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
              5: "driving", 6: "lunch", 7: "walking", 8: "working"}
FEATURE_NAMES = ["mean", "std", "minimum", "maximum", "range", "rms", "median", "iqr",
                 "zero_crossing_rate", "slope", "lag1_autocorrelation", "dominant_frequency",
                 "spectral_entropy", "spectral_centroid", "cardiac_band_fraction"]


def spectrum(x, fs):
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) != 8 * fs or not np.isfinite(x).all():
        raise ValueError("Expected a finite 8-second window")
    # Native rates share a 1/256-Hz grid. Padding is an explicit reconstruction choice.
    magnitude = np.abs(np.fft.rfft(x, n=fs * 256))[:1025]
    return ((magnitude - magnitude.mean()) / max(magnitude.std(), 1e-8)).astype(np.float32)


def descriptors(x, fs):
    x = np.asarray(x, dtype=float).reshape(-1)
    centered = x - x.mean()
    time = np.arange(len(x)) / fs
    time -= time.mean()
    a, b = centered[:-1], centered[1:]
    correlation = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    power = np.abs(np.fft.rfft(centered)) ** 2
    f = np.fft.rfftfreq(len(x), d=1 / fs)
    keep = (f > 0) & (f <= 4)
    p, frequencies = power[keep], f[keep]
    q = p / max(p.sum(), 1e-12)
    positive = q > 0
    entropy = -np.sum(q[positive] * np.log(q[positive])) / np.log(len(q))
    return np.asarray([x.mean(), x.std(), x.min(), x.max(), np.ptp(x), np.sqrt(np.mean(x ** 2)),
        np.median(x), np.percentile(x, 75) - np.percentile(x, 25),
        np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1])),
        np.dot(time, centered) / max(np.dot(time, time), 1e-12), correlation,
        frequencies[p.argmax()] if p.sum() > 1e-12 else 0,
        entropy, np.dot(p, frequencies) / max(p.sum(), 1e-12),
        power[(f >= .5) & (f <= 4)].sum() / max(power.sum(), 1e-12)], dtype=np.float32)


def window_features(bvp, acc):
    bvp, acc = np.asarray(bvp).reshape(-1), np.asarray(acc)
    if bvp.shape != (512,) or acc.shape != (256, 3):
        raise ValueError("Expected BVP [512] and ACC [256,3]")
    if not (np.isfinite(bvp).all() and np.isfinite(acc).all()):
        raise ValueError("Nonfinite sensor samples")
    signals = [(bvp, 64)] + [(acc[:, i], 32) for i in range(3)]
    return (np.stack([spectrum(x, fs) for x, fs in signals]),
            np.concatenate([descriptors(x, fs) for x, fs in signals]))


def get(mapping, key):
    return mapping[key] if key in mapping else mapping[key.encode()]


def prepare_subject(source, destination):
    # Pickle can execute code: only use official, trusted dataset files.
    with open(source, "rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    wrist = get(get(raw, "signal"), "wrist")
    bvp = np.asarray(get(wrist, "BVP"), dtype=float).reshape(-1)
    acc = np.asarray(get(wrist, "ACC"), dtype=float)
    hr = np.asarray(get(raw, "label"), dtype=np.float32).reshape(-1)
    activity = np.asarray(get(raw, "activity"), dtype=float).reshape(-1)
    if not len(hr) or acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("Invalid subject shapes or no HR labels")
    if not all(np.isfinite(x).all() for x in (bvp, acc, hr, activity)):
        raise ValueError("Nonfinite subject data")
    if np.any(activity != np.floor(activity)) or np.any((activity < 0) | (activity > 8)):
        raise ValueError("Activity IDs must be integers 0..8")
    activity = activity.astype(int)
    end = (len(hr) - 1) * 2 + 8
    if len(bvp) < end * 64 or len(acc) < end * 32 or len(activity) < end * 4:
        raise ValueError("Sensor stream ends before final HR window; refusing label misalignment")
    spectra = np.empty((len(hr), 4, 1025), dtype=np.float32)
    features = np.empty((len(hr), 60), dtype=np.float32)
    labels = np.empty(len(hr), dtype=np.int64)
    for i in range(len(hr)):
        start = i * 2
        spectra[i], features[i] = window_features(bvp[start * 64:(start + 8) * 64],
                                                acc[start * 32:(start + 8) * 32])
        labels[i] = np.bincount(activity[start * 4:(start + 8) * 4], minlength=9).argmax()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, spectra=spectra, features=features, hr=hr, activity=labels,
                        start_seconds=np.arange(len(hr)) * 2)
    return len(hr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("cache"))
    args = parser.parse_args()
    files = sorted(p for p in args.data_dir.rglob("S*.pkl") if p.stem[1:].isdigit())
    if not files:
        parser.error("No subject files S1.pkl ... S15.pkl found")
    if len({p.stem for p in files}) != len(files):
        parser.error("Duplicate subject IDs; choose a single dataset root")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        count = prepare_subject(file, args.out_dir / f"{file.stem}.npz")
        print(f"{file.stem}: {count} windows")
    (args.out_dir / "preprocessing.json").write_text(json.dumps(dict(
        window_seconds=8, stride_seconds=2, bvp_hz=64, acc_hz=32, activity_hz=4,
        frequency_bins=1025, frequency_range_hz=[0, 4], fft_grid_hz=1/256,
        channels=["bvp", "acc_x", "acc_y", "acc_z"], features_per_channel=FEATURE_NAMES), indent=2))


if __name__ == "__main__":
    main()

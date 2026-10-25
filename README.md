# Deep PPG: Activity-Aware Heart Rate Estimation

Implementation for **"Deep PPG: Improving Heart Rate Estimates with Activity Prediction"**
Sai Pavan Veluguri — *2025 1st International Conference on AIML-Applications for Engineering & Technology (ICAET)*, IEEE.

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICAET63349.2025.10932185-blue)](https://doi.org/10.1109/ICAET63349.2025.10932185)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



## Idea

Wrist-worn PPG is cheap and continuous, but motion artifacts make heart-rate estimates unreliable during activity. Deep PPG feeds time-frequency spectra of PPG and accelerometer signals into a CNN, and adds a **separate Random Forest that predicts what the wearer is doing**. The predicted activity is fused into the CNN, so the network can adjust for activity-specific motion artifacts.

## Results reported in the paper

On PPG-DaLiA (15 subjects, 8 everyday and sports activities):

| Model | Mean MAE (BPM) |
|---|---|
| Reference Deep PPG CNN (Reiss et al., 2019) | 7.65 ± 4.20 |
| **CNN + predicted activity (this work)** | **7.33 ± 1.15** |

- The activity-aware model matched or beat the reference on most subjects, with much smaller variation between subjects (±1.15 vs ±4.20 BPM).
- The Random Forest activity predictor reached 87–95% accuracy across subjects.
- An untuned XGBoost model on handcrafted features reached 2.30 ± 0.34 BPM, but under a within-subject 80:10:10 split. That setup is not comparable with the cross-subject numbers above.
- The abstract states 7.1 ± 1.3 BPM for the activity-aware CNN; Table I reports 7.33 ± 1.15 BPM.

These figures come from the paper. This repository does not ship results; you produce them by training on the dataset (see below).

## Implemented pipeline

```text
Wrist BVP (64 Hz) + three-axis ACC (32 Hz)
                 |
          8 s windows / 2 s step
                 |
       +---------+----------------------+
       |                                |
  Four FFT spectra               60 window descriptors
  0–4 Hz, 1025 bins                      |
  z-normalized per channel       Random Forest activity model
       |                                |
  Channel + temporal CNN fusion     Predicted probabilities
       |                                |
  Convolution/pooling blocks -------- late fusion
                 |
         Estimated heart rate (BPM)
```

An optional XGBoost regression baseline uses the handcrafted descriptors directly. The CNN uses ELU activations, Adam, and MSE training loss; evaluation reports MAE in BPM.

## Install

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,boost]"
pytest -q
```

For a smaller CPU-only PyTorch install, install it from the official CPU index before installing the project:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The `boost` extra is only needed for XGBoost. GPU support depends on installing a compatible PyTorch build. Dependency ranges are provided, not a frozen reproducibility environment; training saves key package versions in `config.json`.

## Get PPG-DaLiA

Download from the official UCI dataset page:
https://archive.ics.uci.edu/dataset/495/ppg+dalia

Read the official dataset documentation:
https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf

Extract the archive and locate the synchronized subject pickles:

```text
data/PPG_FieldStudy/
├── S1/S1.pkl
├── S2/S2.pkl
...
└── S15/S15.pkl
```

The actual parent folder name can differ; the loader searches recursively for `S*.pkl`. Do not load untrusted pickle or joblib files: both formats can execute code. No automatic download or redistribution of the dataset is included.

```bash
python -m deep_ppg.preprocess --data-dir data/PPG_FieldStudy --out-dir cache
```

BVP and ACC are windowed at their native rates, using the same start time and the supplied ECG-derived HR targets. Activity IDs are majority-voted over the matching 8-second interval of the 4-Hz activity stream; ties choose the lowest ID. Transition class 0 is retained. Trailing sensor samples without a corresponding HR target are ignored. A too-short sensor or activity stream raises an error instead of silently shifting labels. ECG, supplied HR labels, and activity ground truth are never used as sensor predictors.

## Train one held-out fold

```bash
python -m deep_ppg.train \
  --cache-dir cache \
  --test-subject S15 \
  --val-subject S14 \
  --out-dir runs/cnn-S15
```

Subjects S1–S13 are used for training, S14 for validation/early stopping, and S15 only for final evaluation. At least two training subjects plus distinct validation and test subjects are needed for the activity-aware CNN. By default, three consecutive spectra form a causal history: current window plus the preceding two windows. The HR target belongs to the current window. The first two windows of each subject are excluded. A prediction requires the full current 8-second window; it is not instantaneous or zero-latency.

The activity Random Forest is fitted **only on training subjects**. Training inputs to the HR model use subject-grouped out-of-fold activity probabilities: each training subject is predicted by a classifier that did not see that subject. Validation and test activity inputs come from the classifier fitted on all training subjects. This avoids feeding the downstream model in-sample activity guesses. HR target normalization is fitted on training targets only.

A single fold does not establish cross-subject average performance. To run 15 outer folds with a predetermined validation subject:

```bash
for test in $(seq 1 15); do
  val=$((test % 15 + 1))
  python -m deep_ppg.train \
    --cache-dir cache --test-subject S${test} --val-subject S${val} \
    --out-dir runs/cnn-S${test}
done
```

Do not tune hyperparameters after reviewing test-fold outcomes. This fixed-validation-subject protocol is an implementation choice, not a claim to reproduce the paper's experimental split. For a rigorous tuned benchmark, use nested subject-wise validation.

## Run ablations and baseline

CNN without predicted activity, using the same outer split:

```bash
python -m deep_ppg.train --cache-dir cache --test-subject S15 \
  --val-subject S14 --no-activity --out-dir runs/cnn-no-activity-S15
```

XGBoost feature baseline, using the same eligible windows and outer split:

```bash
python -m deep_ppg.train --cache-dir cache --test-subject S15 \
  --val-subject S14 --model xgboost --out-dir runs/xgboost-S15
```

These are alternative models, not an ensemble average of CNN and XGBoost. The paper's word "ensemble" refers here to the activity-model-plus-CNN pipeline. Feature selection, CWT/AR features, SHAP, demographic inputs, and original neural-network ensemble averaging are not implemented.

## Inference

Use an existing cache after training. This command reads only spectra, handcrafted features, and window start times, **not** the subject's true HR or activity labels:

```bash
python -m deep_ppg.predict --run-dir runs/cnn-S15 \
  --subject-cache cache/S15.npz --output runs/cnn-S15/inference.csv
```

The cache-preparation command itself expects the labeled PPG-DaLiA format. For a new unlabeled wearable recording, call `window_features(bvp, acc)` on synchronized native-rate windows, maintain the three-spectrum history, and apply the saved RF, CNN, and target normalization; a raw-device/streaming adapter is not included. Inspect `deep_ppg/predict.py` for the batched inference procedure.

## Outputs

Every successful training run writes:

- `config.json`: exact split, package versions, model settings, and target normalization where applicable
- `metrics.json`: held-out MAE/RMSE, per-activity MAE, and activity accuracy when applicable
- `predictions.csv`: actual held-out predictions and reference values, with window start times
- `training.csv`: CNN epoch loss and validation MAE
- `cnn.pt` and `activity.joblib`, or `xgboost.json`: fitted model artifacts

An existing nonempty run directory is rejected to prevent accidental overwrites. Models and caches are ignored by Git by default. **No accuracy numbers are prefilled.** Results are produced only when you train and evaluate on the actual dataset. Do not commit private participant data. Running all subjects loads cached spectra into RAM; histories are assembled on demand rather than stored repeatedly. Full-dataset runs may require substantial RAM. Training speed and model accuracy have not been benchmarked here.

## Fidelity and deliberate choices

| Component | Paper alignment and limits |
|---|---|
| Inputs | Wrist BVP plus ACC X/Y/Z |
| Segmentation | 8-second windows with 2-second shifts |
| Spectral inputs | Magnitude FFT, 0–4 Hz inclusive, 1025 points, per-window/channel z-normalization |
| FFT grid | New explicit choice: native-rate zero-padding to 1/256 Hz spacing (BVP n=16384; ACC n=8192). Padding does not create additional physical frequency resolution. The supplied paper does not fully specify this padding convention. |
| CNN | Channel fusion, history fusion, seven spectral conv/pool blocks, final compression, two dense layers, ELU, dropout; compact, configurable layer sizes |
| History | Three windows, configurable; default and padding conventions are implementation choices |
| Activity predictor | Random Forest on wrist-only features; nine probabilities with late fusion, instead of the scalar activity input suggested by Figure 3 |
| Features | Fifteen explicitly defined descriptors per channel (60 total), not the paper's exact unreleased feature selection or full feature inventory |
| Training | Adam/MSE; training-only target standardization and validation early stopping are explicit implementation choices |
| Evaluation | Subject-disjoint splits and grouped activity cross-fitting; deliberately different from the paper's reported within-subject 80:10:10 boosted-tree setup |

The uploaded paper reports different final CNN aggregates in its abstract and Table I. This code intentionally adopts neither number as an achieved result. Never compare within-subject/random-window scores directly to this subject-disjoint protocol. Adjacent 8-second windows overlap by six seconds, so randomly splitting windows can share raw observations between sets.

## Repository layout

```text
deep_ppg/
  preprocess.py  # Official pickle loader, alignment, FFTs, descriptors
  activity.py    # Random Forest + grouped out-of-fold probabilities
  model.py       # Activity-aware CNN
  data.py        # Cached windows and causal histories
  train.py       # Training, ablation, baseline, metrics
  predict.py     # Saved-model inference
tests/           # Unit checks, not a clinical/accuracy benchmark
```

## References and publishing

Associated paper: https://doi.org/10.1109/ICAET63349.2025.10932185

Dataset: Reiss, Indlekofer, and Schmidt (2019), PPG-DaLiA, https://doi.org/10.24432/C53890.

Original Deep PPG methodology: Reiss et al. (2019), *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks*, https://pubmed.ncbi.nlm.nih.gov/31336894/.

## Citation

```bibtex
@inproceedings{veluguri2025deepppg,
  author    = {Veluguri, Sai Pavan},
  title     = {Deep PPG: Improving Heart Rate Estimates with Activity Prediction},
  booktitle = {2025 1st International Conference on AIML-Applications for Engineering \& Technology (ICAET)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICAET63349.2025.10932185}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper is © IEEE and is not bundled here; read it on [IEEE Xplore](https://doi.org/10.1109/ICAET63349.2025.10932185).

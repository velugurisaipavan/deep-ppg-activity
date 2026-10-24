"""Subject-grouped cross-fitting avoids in-sample activity predictions for HR training."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


def classifier(seed=42, trees=200):
    return RandomForestClassifier(n_estimators=trees, max_depth=18, min_samples_leaf=2,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def probabilities(model, features):
    result = np.zeros((len(features), 9), dtype=np.float32)
    result[:, model.classes_.astype(int)] = model.predict_proba(features)
    return result


def fit_activity(features, labels, groups, seed=42, trees=200):
    count = len(np.unique(groups))
    if count < 2:
        raise ValueError("Need at least two training subjects for grouped cross-fitting")
    out_of_fold = np.zeros((len(features), 9), dtype=np.float32)
    splitter = GroupKFold(n_splits=min(5, count))
    for train, held_out in splitter.split(features, labels, groups):
        model = classifier(seed, trees).fit(features[train], labels[train])
        out_of_fold[held_out] = probabilities(model, features[held_out])
    model = classifier(seed, trees).fit(features, labels)
    return model, out_of_fold

import time
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Calculates the AUROC score
def compute_auroc(y_true, y_proba, model):
    present_classes = np.unique(y_true)
    if len(present_classes) < 2:
        return np.nan
    if len(present_classes) == 2:
        pos_idx = list(model.classes_).index(present_classes.max())
        return roc_auc_score(y_true, y_proba[:, pos_idx])
    mask = np.isin(model.classes_, present_classes)
    return roc_auc_score(
        y_true, y_proba[:, mask], labels=present_classes, multi_class="ovr", average="macro"
    )

# Calculates Jaccard score for feature stability
def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    if len(a_set | b_set) == 0:
        return np.nan
    return len(a_set & b_set) / len(a_set | b_set)

# 95% confidence interval margin
def compute_confidence_interval(data, confidence=0.95):
    clean_data = [x for x in data if not np.isnan(x)]
    if len(clean_data) < 2:
        return 0.0
    sem = stats.sem(clean_data)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, len(clean_data) - 1)
    return margin

# main validation function. Default: 5 outer CVs and 3 inner CVs
def nested_cv_evaluation(X, y, selector_fn, param_grid, selector_name, outer_splits=5, inner_splits=3, return_grid_scores=False):
    """
    Executes Nested Stratified K-Fold CV.
    If return_grid_scores is True, records mean inner-CV validation scores across all folds
    for each hyperparameter combination to enable unbiased sensitivity plots.
    """
    outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)

    # Information about outer CVs
    outer_accs, outer_bal_accs, outer_aurocs = [], [], []
    runtimes, selected_features_all, n_features_all = [], [], []
    best_params_per_fold = []

    # Matrix to store inner-CV scores across folds. shape is (outer_splits, len(param_grid))
    grid_scores_matrix = [] if return_grid_scores else None

    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, y)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        best_param = None
        best_inner_score = -1.0
        fold_grid_scores = []

        # Inner CV: Hyperparameter combinations
        if param_grid and len(param_grid) > 1:
            inner_kf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42 + fold)

            for param_num, params in enumerate(param_grid, start=1):
                inner_scores, inner_features, inner_runtimes = [], [], []
                
                for inner_fold, (in_train_idx, in_val_idx) in enumerate(
                    inner_kf.split(X_train_outer, y_train_outer)
                ):
                    print(f"{selector_name}: Outer fold {fold + 1}/{outer_splits} | Param set {param_num}/{len(param_grid)} | inner fold {inner_fold + 1}/{inner_splits}")
                    X_tr_in, X_val_in = X_train_outer.iloc[in_train_idx], X_train_outer.iloc[in_val_idx]
                    y_tr_in, y_val_in = y_train_outer[in_train_idx], y_train_outer[in_val_idx]

                    p = dict(params)
                    p["seed"] = 42 + fold
                    # At the last outer CV split, stores the plot (only applicable for DyGraFS)
                    p["save_fig"] = (fold + 1 == outer_splits)
                    start_inner_time = time.time()

                    selected_in = selector_fn(X_tr_in, y_tr_in, p) if selector_fn else X_tr_in.columns.tolist()
                    selected_in = list(set(selected_in).intersection(X_tr_in.columns))
                    inner_runtime = time.time() - start_inner_time
                    inner_runtimes.append(inner_runtime)

                    # No feature selected
                    if len(selected_in) == 0:
                        inner_scores.append(0.0)
                        inner_features.append(0)
                        continue

                    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
                    clf.fit(X_tr_in[selected_in], y_tr_in)
                    preds = clf.predict(X_val_in[selected_in])
                    inner_scores.append(balanced_accuracy_score(y_val_in, preds))
                    inner_features.append(len(selected_in))

                mean_in_score = np.nanmean(inner_scores)
                mean_in_feat = np.nanmean(inner_features)

                # Specifically for DyGraFS, we are also storing inner CV performance to plot additional figures
                if return_grid_scores:
                    mean_in_runtime = np.nanmean(inner_runtimes)
                    fold_grid_scores.append({
                        "balanced_accuracy_mean": mean_in_score,
                        "features_mean": mean_in_feat,
                        "runtime_mean": mean_in_runtime
                    })

                if mean_in_score > best_inner_score:
                    # Updates the best set of parameters
                    best_inner_score = mean_in_score
                    best_param = params

        else:
            # Only 1 parameters combination available, no need to iterate
            best_param = param_grid[0] if param_grid else {}
            if return_grid_scores:
                fold_grid_scores.append({
                    "balanced_accuracy_mean": np.nan,
                    "features_mean": X_train_outer.shape[1] if selector_fn is None else np.nan,
                    "runtime_mean": np.nan
                })

        best_params_per_fold.append(best_param)

        # Outer loop: Outer evaluation based in the best param defined
        # (params used in the outer CV are based in the best inner CV param combination)
        p_outer = dict(best_param) if best_param else {}
        p_outer["seed"] = 42 + fold
        p_outer["save_fig"] = False

        start_time = time.time()
        selected_outer = selector_fn(X_train_outer, y_train_outer, p_outer) if selector_fn else X_train_outer.columns.tolist()
        selected_outer = list(set(selected_outer).intersection(X_train_outer.columns))
        runtime = time.time() - start_time

        if len(selected_outer) == 0:
            # No features selected (which is an issue)
            outer_accs.append(np.nan)
            outer_bal_accs.append(np.nan)
            outer_aurocs.append(np.nan)
            runtimes.append(runtime)
            selected_features_all.append([])
            n_features_all.append(0)
            if return_grid_scores:
                grid_scores_matrix.append(fold_grid_scores)
            continue

        clf_outer = RandomForestClassifier(n_estimators=200, random_state=p_outer["seed"], class_weight="balanced")
        clf_outer.fit(X_train_outer[selected_outer], y_train_outer)
        y_pred = clf_outer.predict(X_test_outer[selected_outer])
        y_proba = clf_outer.predict_proba(X_test_outer[selected_outer])

        # Stores the results of the current outer CV fold
        outer_accs.append(accuracy_score(y_test_outer, y_pred))
        outer_bal_accs.append(balanced_accuracy_score(y_test_outer, y_pred))
        outer_aurocs.append(compute_auroc(y_test_outer, y_proba, clf_outer))
        runtimes.append(runtime if selector_fn else 0)
        selected_features_all.append(selected_outer)
        n_features_all.append(len(selected_outer))

        # Append inner fold results to grid_scores_matrix
        if return_grid_scores:
            grid_scores_matrix.append(fold_grid_scores)

    # Feature Stability across Outer Folds
    if len(selected_features_all) > 1:
        stability = np.nanmean([jaccard(a, b) for a, b in combinations(selected_features_all, 2)])
    else:
        stability = np.nan

    # Feature selection frequency across OUTER CV folds (This measures how often each feature was selected by the
    # complete nested-CV procedure in an unbiased outer fold)
    feature_selection_frequency = {}

    for fold_features in selected_features_all:
        for feature in fold_features:
            feature_selection_frequency[feature] = (
                feature_selection_frequency.get(feature, 0) + 1
            )

    n_valid_outer_folds = len(selected_features_all)

    # Store both absolute count and frequency percent
    feature_selection_frequency_pct = {
        feature: (count / n_valid_outer_folds) * 100
        for feature, count in feature_selection_frequency.items()
    }

    # Rank features primarily by frequency, then alphabetically order
    feature_selection_frequency_ranked = sorted(
        feature_selection_frequency.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Aggregate Inner Grid Scores across Outer Folds
    # NOTE: innter grid summary is a side benchmark to evaluate DyGraFS interactions across 
    # parameters. It is not reliable for comparing selectors
    inner_grid_summary = None
    if return_grid_scores and param_grid:
        inner_summary_list = []

        for i, params in enumerate(param_grid):
            p_dict = dict(params)

            # One value per OUTER fold, where each value is already the mean selector runtime across that fold's INNER folds.
            accs = [
                grid_scores_matrix[f][i]["balanced_accuracy_mean"]
                for f in range(outer_splits)
            ]

            feats = [
                grid_scores_matrix[f][i]["features_mean"]
                for f in range(outer_splits)
            ]

            inner_runtimes = [
                grid_scores_matrix[f][i]["runtime_mean"]
                for f in range(outer_splits)
            ]

            p_dict["selector"] = selector_name
            p_dict["balanced_accuracy_mean"] = np.nanmean(accs)
            p_dict["features_mean"] = np.nanmean(feats)
            p_dict["runtime_mean"] = np.nanmean(inner_runtimes)

            inner_summary_list.append(p_dict)

        inner_grid_summary = pd.DataFrame(inner_summary_list)

    # Returns selector results for outer CV + optional inner CV data
    # This will be replicated for each selector tested
    return {
        "selector": selector_name,
        "accuracy_mean": np.nanmean(outer_accs),
        "accuracy_ci95": compute_confidence_interval(outer_accs),
        "balanced_accuracy_mean": np.nanmean(outer_bal_accs),
        "balanced_accuracy_std": np.nanstd(outer_bal_accs),
        "balanced_accuracy_ci95": compute_confidence_interval(outer_bal_accs),
        "outer_bal_accs_folds": outer_bal_accs,
        "auroc_mean": np.nanmean(outer_aurocs),
        "auroc_ci95": compute_confidence_interval(outer_aurocs),
        "feature_stability": stability,
        "runtime_mean": np.nanmean(runtimes),
        "features_mean": np.nanmean(n_features_all),
        "selected_features_outer_folds": selected_features_all,
        "feature_selection_frequency": feature_selection_frequency,
        "feature_selection_frequency_pct": feature_selection_frequency_pct,
        "feature_selection_frequency_ranked": feature_selection_frequency_ranked,
        "best_params_per_fold": best_params_per_fold,
        "inner_grid_summary": inner_grid_summary
    }

# Support funcion for the paired wilcoxon tests (DyGraFS vs each other selector)
def holm_bonferroni_correction(p_values):
    """
    Step-down Holm-Bonferroni correction for family-wise error rate control.
    More powerful than plain Bonferroni while still controlling FWER without
    assuming independence between tests (valid here, since paired Wilcoxon
    tests on shared CV folds are not independent of each other).

    NaNs are passed through unchanged and excluded from the ranking/family size.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    corrected = np.full(n, np.nan)

    valid_idx = np.where(~np.isnan(p_values))[0]
    m = len(valid_idx)
    if m == 0:
        return corrected

    order = valid_idx[np.argsort(p_values[valid_idx])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)

    return corrected


def run_paired_wilcoxon_tests(df_results):
    """
    Diagnostic comparison of every selector against DyGraFS specifically
    (paired Wilcoxon signed-rank test on matched outer-CV fold scores).

    NOTE: this view only tests selectors against DyGraFS, not every possible
    pair (e.g. classical-vs-classical, or vs. GFSIR). It is kept for quick,
    DyGraFS-centric reporting, but a Holm-Bonferroni correction is applied
    across this subfamily of tests
    """
    dygrafs_row = df_results[df_results["selector"] == "DyGraFS"]
    if dygrafs_row.empty:
        return df_results

    dygrafs_scores = dygrafs_row.iloc[0]["outer_bal_accs_folds"]
    p_values = []

    for idx, row in df_results.iterrows():
        scores = row["outer_bal_accs_folds"]
        if scores is None or np.array_equal(scores, dygrafs_scores):
            p_values.append(1.0)
            continue
        try:
            _, p_val = stats.wilcoxon(scores, dygrafs_scores)
            p_values.append(p_val)
        except Exception:
            p_values.append(np.nan)

    df_results["p_value_vs_dygrafs"] = p_values
    df_results["p_value_vs_dygrafs_holm"] = holm_bonferroni_correction(p_values)
    return df_results

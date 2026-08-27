import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

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

def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    if len(a_set | b_set) == 0:
        return np.nan
    return len(a_set & b_set) / len(a_set | b_set)

def compute_confidence_interval(data, confidence=0.95):
    clean_data = [x for x in data if not np.isnan(x)]
    if len(clean_data) < 2:
        return 0.0
    sem = stats.sem(clean_data)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, len(clean_data) - 1)
    return margin

def nested_cv_evaluation(X, y, selector_fn, param_grid, selector_name, outer_splits=5, inner_splits=3, return_grid_scores=False):
    """
    Executes Nested Stratified K-Fold CV.
    If return_grid_scores is True, records mean inner-CV validation scores across all folds
    for each hyperparameter combination to enable unbiased sensitivity plots.
    """
    outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    
    outer_accs, outer_bal_accs, outer_aurocs = [], [], []
    runtimes, selected_features_all, n_features_all = [], [], []
    best_params_per_fold = []

    # Matrix to store inner-CV scores across folds: shape (outer_splits, len(param_grid))
    grid_scores_matrix = [] if return_grid_scores else None

    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, y)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        best_param = None
        best_inner_score = -1.0
        fold_grid_scores = []

        # --- INNER LOOP: Hyperparameter Selection ---
        if param_grid and len(param_grid) > 1:
            inner_kf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42 + fold)
            for params in param_grid:
                inner_scores = []
                inner_features = []
                
                for in_train_idx, in_val_idx in inner_kf.split(X_train_outer, y_train_outer):
                    X_tr_in, X_val_in = X_train_outer.iloc[in_train_idx], X_train_outer.iloc[in_val_idx]
                    y_tr_in, y_val_in = y_train_outer[in_train_idx], y_train_outer[in_val_idx]

                    p = dict(params)
                    p["seed"] = 42 + fold
                    p["save_fig"] = False
                    
                    selected_in = selector_fn(X_tr_in, y_tr_in, p) if selector_fn else X_tr_in.columns.tolist()
                    selected_in = list(set(selected_in).intersection(X_tr_in.columns))

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

                if return_grid_scores:
                    fold_grid_scores.append({
                        "balanced_accuracy_mean": mean_in_score,
                        "features_mean": mean_in_feat
                    })

                if mean_in_score > best_inner_score:
                    best_inner_score = mean_in_score
                    best_param = params
        else:
            best_param = param_grid[0] if param_grid else {}

        if return_grid_scores:
            grid_scores_matrix.append(fold_grid_scores)

        best_params_per_fold.append(best_param)

        # --- OUTER LOOP: Outer Evaluation ---
        start_time = time.time()
        p_outer = dict(best_param) if best_param else {}
        p_outer["seed"] = 42 + fold
        p_outer["save_fig"] = (fold == 0)

        selected_outer = selector_fn(X_train_outer, y_train_outer, p_outer) if selector_fn else X_train_outer.columns.tolist()
        selected_outer = list(set(selected_outer).intersection(X_train_outer.columns))
        runtime = time.time() - start_time

        if len(selected_outer) == 0:
            outer_accs.append(np.nan)
            outer_bal_accs.append(np.nan)
            outer_aurocs.append(np.nan)
            runtimes.append(runtime)
            selected_features_all.append([])
            n_features_all.append(0)
            continue

        clf_outer = RandomForestClassifier(n_estimators=200, random_state=42 + fold, class_weight="balanced")
        clf_outer.fit(X_train_outer[selected_outer], y_train_outer)
        y_pred = clf_outer.predict(X_test_outer[selected_outer])
        y_proba = clf_outer.predict_proba(X_test_outer[selected_outer])

        outer_accs.append(accuracy_score(y_test_outer, y_pred))
        outer_bal_accs.append(balanced_accuracy_score(y_test_outer, y_pred))
        outer_aurocs.append(compute_auroc(y_test_outer, y_proba, clf_outer))
        runtimes.append(runtime if selector_fn else 0)
        selected_features_all.append(selected_outer)
        n_features_all.append(len(selected_outer))

    # Feature Stability across Outer Folds
    if len(selected_features_all) > 1:
        stability = np.nanmean([jaccard(a, b) for a, b in combinations(selected_features_all, 2)])
    else:
        stability = np.nan

    # Aggregate Inner Grid Scores across Outer Folds
    inner_grid_summary = None
    if return_grid_scores and param_grid:
        inner_summary_list = []
        for i, params in enumerate(param_grid):
            p_dict = dict(params)
            accs = [grid_scores_matrix[f][i]["balanced_accuracy_mean"] for f in range(outer_splits)]
            feats = [grid_scores_matrix[f][i]["features_mean"] for f in range(outer_splits)]
            
            p_dict["selector"] = selector_name
            p_dict["balanced_accuracy_mean"] = np.nanmean(accs)
            p_dict["features_mean"] = np.nanmean(feats)
            p_dict["runtime_mean"] = np.nanmean(runtimes) # Approximate runtimes
            inner_summary_list.append(p_dict)
            
        inner_grid_summary = pd.DataFrame(inner_summary_list)

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
        "best_params_per_fold": best_params_per_fold,
        "inner_grid_summary": inner_grid_summary
    }

def run_paired_wilcoxon_tests(df_results):
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
    return df_results
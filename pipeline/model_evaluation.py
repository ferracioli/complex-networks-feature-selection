import itertools
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from boruta import BorutaPy

from pipeline.feature_selector import select_cn_centers
from pipeline.GFSIR.graph_feature_selection import GraphFeatureSelection
import pipeline.model_plots as plots
import pipeline.model_plots_pt as plots_pt
from pipeline.nested_cv import nested_cv_evaluation, run_paired_wilcoxon_tests

warnings.filterwarnings("ignore")
np.random.seed(42)

with open('input/config.json', 'r') as file:
    config = json.load(file)

# --- SELECTOR IMPLEMENTATIONS ---

def variance_selector(X_train, y_train, params=None):
    th = params.get("threshold", 1e-5) if params else 1e-5
    vt = VarianceThreshold(threshold=th)
    vt.fit(X_train)
    return X_train.columns[vt.get_support()].tolist()

def anova_selector(X_train, y_train, params=None):
    percentile = params.get("percentile", 50) if params else 50
    scores, _ = f_classif(X_train, y_train)
    scores = np.nan_to_num(scores, nan=0.0)
    threshold = np.percentile(scores, percentile)
    return X_train.columns[scores >= threshold].tolist()

def mi_selector(X_train, y_train, params=None):
    n_neighbors = params.get("n_neighbors", 3) if params else 3
    percentile = params.get("percentile", 50) if params else 50
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    scores = mutual_info_classif(X_scaled, y_train, n_neighbors=n_neighbors, random_state=42)
    threshold = np.percentile(scores, percentile)
    return X_train.columns[scores >= threshold].tolist()

def l1logistic_selector(X_train, y_train, params=None):
    c_val = params.get("C", 1.0) if params else 1.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(penalty="l1", solver="saga", C=c_val, class_weight="balanced", max_iter=5000, random_state=42)
    model.fit(X_scaled, y_train)
    coef = np.abs(model.coef_).sum(axis=0)
    return X_train.columns[coef > 1e-6].tolist()

def rfe_selector(X_train, y_train, params=None):
    ratio = params.get("feature_ratio", 0.5) if params else 0.5
    c_val = params.get("C", 1.0) if params else 1.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    svc = SVC(kernel="linear", C=c_val, random_state=42)
    n_features = max(1, int(X_train.shape[1] * ratio))
    rfe = RFE(estimator=svc, n_features_to_select=n_features)
    rfe.fit(X_scaled, y_train)
    return X_train.columns[rfe.support_].tolist()

def boruta_selector(X_train, y_train, params=None):
    perc = params.get("perc", 100) if params else 100
    max_iter = params.get("max_iter", 100) if params else 100
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    boruta = BorutaPy(estimator=rf, n_estimators="auto", perc=perc, max_iter=max_iter, random_state=42, verbose=0)
    y_array = y_train.values if hasattr(y_train, "values") else y_train
    boruta.fit(X_scaled, y_array)
    return X_train.columns[boruta.support_].tolist()

def gfsir_grid(X_train, y_train, params=None):
    assert params is not None
    selector = GraphFeatureSelection(
        input_dir=".", output_dir=".", lower_threshold=params["gfsir_minth"],
        upper_threshold=params["gfsir_maxth"], n_features=params["gfsir_nfeatures"]
    )
    if params["gfsir_minth"] == "auto":
        df_selected = selector.apply_graph_feature_selection(X_train.copy(), method=params["gfsir_selector"], mode="adaptive")
    else:
        df_selected = selector.apply_graph_feature_selection(X_train.copy(), method=params["gfsir_selector"], mode="manual")
    return df_selected.columns.tolist()

def graph_selector(X_train, y_train, params):
    image_filename = f"{params['dataset']}_{params['similarity_function']}_{params['threshold']:.2f}_{params['cn_selector']}_radiomic_graph.png"
    return select_cn_centers(
        X_train, threshold=params["threshold"], cn_selector=params["cn_selector"],
        similarity_function=params["similarity_function"], seed_nb=params["seed"],
        save_fig=params["save_fig"], png_path=f"outputs/{params['dataset']}/feature_plots/{image_filename}"
    )

# --- HELPER FUNCTION FOR EVALUATION RUNS ---

def run_selector_evaluation(X, y, selector_fn, param_grid, name, outer_splits, return_grid_scores=False):
    num_combinations = len(param_grid) if param_grid else 1
    print(f" -> Running {name} ({num_combinations} combination(s))...", flush=True)
    
    t0 = time.time()
    res = nested_cv_evaluation(X, y, selector_fn, param_grid, name, outer_splits=outer_splits, return_grid_scores=return_grid_scores)
    elapsed = time.time() - t0
    
    bal_acc = res.get("balanced_accuracy_mean", np.nan)
    feats = res.get("features_mean", np.nan)
    print(f"    Finished {name} in {elapsed:.2f}s | Mean Bal. Acc: {bal_acc:.4f} | Avg Features: {feats:.1f}", flush=True)
    
    return res, elapsed, num_combinations

# --- BENCHMARK EXECUTION ---

def model_benchmarking(dataset="sample"):
    start_total_time = time.time()
    print("=" * 70)
    print(f"STARTING BENCHMARK FOR DATASET: '{dataset}'")
    print("=" * 70, flush=True)

    radiomic_features_path = f"{config[dataset]['output_path']}{dataset}_radiomic_features.csv"
    df = pd.read_csv(radiomic_features_path)
    tg_column = config[dataset]["target_column"]
    df = df.dropna(subset=[tg_column])

    if "drop_rare_classes" in dataset:
        class_counts = df[tg_column].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        df = df[df[tg_column].isin(valid_classes)]

    static_remove = [tg_column, "exam_path", "gt_path", "patient_id"]
    dynamic_remove = config[dataset].get("to_remove_columns", [])
    X = df.drop(columns=static_remove + dynamic_remove, errors="ignore")
    le = LabelEncoder()
    y = np.asarray(le.fit_transform(df[tg_column]))

    min_class_size = np.min(np.bincount(y))
    outer_splits = min(5, min_class_size)
    print(f"Dataset Loaded | Shape: {X.shape} | Outer Splits: {outer_splits}", flush=True)

    results = []
    selector_stats = []
    grid_cfg = config[dataset]["grid_params"]

    print("\n--- Running Feature Selectors ---", flush=True)

    # 1. Vanilla RF Baseline
    res, elapsed, n_combs = run_selector_evaluation(X, y, None, [{}], "Vanilla RF", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "Vanilla RF", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # 2. DyGraFS
    dygrafs_grid = [
        {"dataset": dataset, "threshold": th, "cn_selector": cn, "similarity_function": sim}
        for sim, th, cn in itertools.product(
            grid_cfg['similarity_functions'],
            grid_cfg['thresholds'],
            grid_cfg['cn_selectors']
        )
    ]
    dygrafs_res, elapsed, n_combs = run_selector_evaluation(
        X, y, graph_selector, dygrafs_grid, "DyGraFS", outer_splits, return_grid_scores=True
    )
    results.append(dygrafs_res)
    selector_stats.append({"Selector": "DyGraFS", "Combinations": n_combs, "Runtime_Sec": elapsed})

    dygrafs_inner_summary = dygrafs_res.pop("inner_grid_summary")
    os.makedirs(f"outputs/{dataset}", exist_ok=True)
    dygrafs_inner_summary.to_csv(f"outputs/{dataset}/{dataset}_dygrafs_inner_grid.csv", index=False)

    # 3. Classical Selectors (Grids pulled from config.json)
    
    # Variance
    variance_grid = [{"threshold": t} for t in grid_cfg["variance_thresholds"]]
    res, elapsed, n_combs = run_selector_evaluation(X, y, variance_selector, variance_grid, "Variance", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "Variance", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # ANOVA
    anova_grid = [{"percentile": p} for p in grid_cfg["anova_percentiles"]]
    res, elapsed, n_combs = run_selector_evaluation(X, y, anova_selector, anova_grid, "Anova", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "Anova", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # Mutual Information
    mi_grid = [
        {"percentile": p, "n_neighbors": k}
        for p, k in itertools.product(grid_cfg["mi_percentiles"], grid_cfg["mi_n_neighbors"])
    ]
    res, elapsed, n_combs = run_selector_evaluation(X, y, mi_selector, mi_grid, "Mutual Information", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "Mutual Information", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # L1 Logistic Regression
    l1_grid = [{"C": c} for c in grid_cfg["l1_c"]]
    res, elapsed, n_combs = run_selector_evaluation(X, y, l1logistic_selector, l1_grid, "L1 Logistic Regression", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "L1 Logistic Regression", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # RFE (SVM)
    rfe_grid = [
        {"feature_ratio": r, "C": c}
        for r, c in itertools.product(grid_cfg["rfe_ratios"], grid_cfg["rfe_c"])
    ]
    res, elapsed, n_combs = run_selector_evaluation(X, y, rfe_selector, rfe_grid, "RFE (SVM)", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "RFE (SVM)", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # Boruta
    boruta_grid = [
        {"perc": p, "max_iter": m}
        for p, m in itertools.product(grid_cfg["boruta_perc"], grid_cfg["boruta_max_iter"])
    ]
    res, elapsed, n_combs = run_selector_evaluation(X, y, boruta_selector, boruta_grid, "Boruta", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "Boruta", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # 4. GFSIR Grid
    gfsir_grid_params = [
        {"gfsir_nfeatures": nf, "gfsir_minth": minth, "gfsir_maxth": maxth, "gfsir_selector": sel}
        for nf, minth, maxth, sel in itertools.product(
            grid_cfg['gfsir_nfeatures'],
            grid_cfg['gfsir_minth'],
            grid_cfg['gfsir_maxth'],
            grid_cfg['gfsir_selector']
        )
    ]
    res, elapsed, n_combs = run_selector_evaluation(X, y, gfsir_grid, gfsir_grid_params, "GFSIR", outer_splits)
    results.append(res)
    selector_stats.append({"Selector": "GFSIR", "Combinations": n_combs, "Runtime_Sec": elapsed})

    # Output Summary & Stats
    summary = pd.DataFrame(results)
    summary = run_paired_wilcoxon_tests(summary)
    summary.to_csv(f"outputs/{dataset}/{dataset}_benchmark_results.csv", index=False)

    total_time = time.time() - start_total_time

    # Write summary TXT file
    txt_path = f"outputs/{dataset}/{dataset}_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"BENCHMARK OVERALL SUMMARY: DATASET '{dataset}'\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Execution Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")
        f.write(f"Dataset Dimensions: {X.shape[0]} samples, {X.shape[1]} features\n")
        f.write(f"Outer CV Splits: {outer_splits}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Selector':<25} | {'Combinations':<12} | {'Time (s)':<10}\n")
        f.write("-" * 60 + "\n")
        for stat in selector_stats:
            f.write(f"{stat['Selector']:<25} | {stat['Combinations']:<12} | {stat['Runtime_Sec']:<10.2f}\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("PERFORMANCE RESULTS:\n")
        f.write("-" * 60 + "\n")
        for res_item in results:
            sel = res_item['selector']
            acc = res_item['balanced_accuracy_mean']
            feats = res_item['features_mean']
            f.write(f"Selector: {sel:<20} | Mean Bal Acc: {acc:.4f} | Avg Features: {feats:.2f}\n")

    print(f"\nSaved run summary text file to: {txt_path}")
    print(f"TOTAL RUNTIME FOR '{dataset}': {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 70, flush=True)

    # --- PLOTTING PIPELINE ---
    plots.performance_boxplot(summary, dataset, metric="balanced_accuracy")
    plots.feature_stability_plot(summary, dataset)
    plots_pt.performance_boxplot_pt(summary, dataset, metric="balanced_accuracy")

    if dygrafs_inner_summary is not None and len(dygrafs_inner_summary) > 0:
        grid_summary_full = pd.concat([summary, dygrafs_inner_summary], ignore_index=True)
        plots.accuracy_vs_runtime_by_threshold(grid_summary_full, dataset)
        plots.accuracy_vs_runtime_by_similarity_function(grid_summary_full, dataset)
        plots.accuracy_vs_runtime_by_cn_selector(grid_summary_full, dataset)
        plots.accuracy_vs_features_by_threshold(grid_summary_full, dataset)
        plots.accuracy_vs_features_by_similarity_function(grid_summary_full, dataset)
        plots.accuracy_vs_features_by_cn_selector(grid_summary_full, dataset)
        plots.accuracy_vs_threshold_by_cn_selector(grid_summary_full, dataset)

# import itertools
# import json
# import warnings
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import mutual_info_classif, RFE, VarianceThreshold, f_classif
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from boruta import BorutaPy
# from pipeline.feature_selector import select_cn_centers
# from pipeline.GFSIR.graph_feature_selection import GraphFeatureSelection
# import pipeline.model_plots as plots
# import pipeline.model_plots_pt as plots_pt
# from pipeline.nested_cv import nested_cv_evaluation, run_paired_wilcoxon_tests

# warnings.filterwarnings("ignore")
# np.random.seed(42)

# with open('input/config.json', 'r') as file:
#     config = json.load(file)

# # --- SELECTOR IMPLEMENTATIONS ---

# def variance_selector(X_train, y_train, params=None):
#     th = params.get("threshold", 1e-5) if params else 1e-5
#     vt = VarianceThreshold(threshold=th)
#     vt.fit(X_train)
#     return X_train.columns[vt.get_support()].tolist()

# def anova_selector(X_train, y_train, params=None):
#     percentile = params.get("percentile", 50) if params else 50
#     scores, _ = f_classif(X_train, y_train)
#     scores = np.nan_to_num(scores, nan=0.0)
#     threshold = np.percentile(scores, percentile)
#     return X_train.columns[scores >= threshold].tolist()

# def mi_selector(X_train, y_train, params=None):
#     n_neighbors = params.get("n_neighbors", 3) if params else 3
#     percentile = params.get("percentile", 50) if params else 50
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     scores = mutual_info_classif(X_scaled, y_train, n_neighbors=n_neighbors, random_state=42)
#     threshold = np.percentile(scores, percentile)
#     return X_train.columns[scores >= threshold].tolist()

# def l1logistic_selector(X_train, y_train, params=None):
#     c_val = params.get("C", 1.0) if params else 1.0
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     model = LogisticRegression(penalty="l1", solver="saga", C=c_val, class_weight="balanced", max_iter=5000, random_state=42)
#     model.fit(X_scaled, y_train)
#     coef = np.abs(model.coef_).sum(axis=0)
#     return X_train.columns[coef > 1e-6].tolist()

# def rfe_selector(X_train, y_train, params=None):
#     ratio = params.get("feature_ratio", 0.5) if params else 0.5
#     c_val = params.get("C", 1.0) if params else 1.0
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     svc = SVC(kernel="linear", C=c_val, random_state=42)
#     n_features = max(1, int(X_train.shape[1] * ratio))
#     rfe = RFE(estimator=svc, n_features_to_select=n_features)
#     rfe.fit(X_scaled, y_train)
#     return X_train.columns[rfe.support_].tolist()

# def boruta_selector(X_train, y_train, params=None):
#     perc = params.get("perc", 100) if params else 100
#     max_iter = params.get("max_iter", 100) if params else 100
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
#     boruta = BorutaPy(estimator=rf, n_estimators="auto", perc=perc, max_iter=max_iter, random_state=42, verbose=0)
#     y_array = y_train.values if hasattr(y_train, "values") else y_train
#     boruta.fit(X_scaled, y_array)
#     return X_train.columns[boruta.support_].tolist()

# def gfsir_grid(X_train, y_train, params=None):
#     assert params is not None
#     selector = GraphFeatureSelection(
#         input_dir=".", output_dir=".", lower_threshold=params["gfsir_minth"],
#         upper_threshold=params["gfsir_maxth"], n_features=params["gfsir_nfeatures"]
#     )
#     if params["gfsir_minth"] == "auto":
#         df_selected = selector.apply_graph_feature_selection(X_train.copy(), method=params["gfsir_selector"], mode="adaptive")
#     else:
#         df_selected = selector.apply_graph_feature_selection(X_train.copy(), method=params["gfsir_selector"], mode="manual")
#     return df_selected.columns.tolist()

# def graph_selector(X_train, y_train, params):
#     image_filename = f"{params['dataset']}_{params['similarity_function']}_{params['threshold']:.2f}_{params['cn_selector']}_radiomic_graph.png"
#     return select_cn_centers(
#         X_train, threshold=params["threshold"], cn_selector=params["cn_selector"],
#         similarity_function=params["similarity_function"], seed_nb=params["seed"],
#         save_fig=params["save_fig"], png_path=f"outputs/{params['dataset']}/feature_plots/{image_filename}"
#     )

# # --- BENCHMARK EXECUTION ---

# def model_benchmarking(dataset="sample"):
#     radiomic_features_path = f"{config[dataset]['output_path']}{dataset}_radiomic_features.csv"
#     df = pd.read_csv(radiomic_features_path)
#     tg_column = config[dataset]["target_column"]
#     df = df.dropna(subset=[tg_column])

#     if "drop_rare_classes" in dataset:
#         class_counts = df[tg_column].value_counts()
#         valid_classes = class_counts[class_counts >= 5].index
#         df = df[df[tg_column].isin(valid_classes)]

#     static_remove = [tg_column, "exam_path", "gt_path", "patient_id"]
#     dynamic_remove = config[dataset].get("to_remove_columns", [])
#     X = df.drop(columns=static_remove + dynamic_remove, errors="ignore")
#     le = LabelEncoder()
#     y = np.asarray(le.fit_transform(df[tg_column]))

#     min_class_size = np.min(np.bincount(y))
#     outer_splits = min(5, min_class_size)
#     print("Final number of splits:", outer_splits)

#     results = []

#     # 1. Vanilla RF Baseline
#     results.append(nested_cv_evaluation(X, y, None, [{}], "Vanilla RF", outer_splits=outer_splits))

#     # 2. DyGraFS (Grid Search evaluated via Inner CV & Inner grid score extraction)
#     dygrafs_grid = [
#         {"dataset": dataset, "threshold": th, "cn_selector": cn, "similarity_function": sim}
#         for sim, th, cn in itertools.product(
#             config[dataset]['grid_params']['similarity_functions'],
#             config[dataset]['grid_params']['thresholds'],
#             config[dataset]['grid_params']['cn_selectors']
#         )
#     ]
#     dygrafs_res = nested_cv_evaluation(
#         X, y, graph_selector, dygrafs_grid, "DyGraFS", outer_splits=outer_splits, return_grid_scores=True
#     )
#     results.append(dygrafs_res)
    
#     # Save DyGraFS Inner Grid Summary for sensitivity analysis
#     dygrafs_inner_summary = dygrafs_res.pop("inner_grid_summary")
#     dygrafs_inner_summary.to_csv(f"outputs/{dataset}/{dataset}_dygrafs_inner_grid.csv", index=False)

#     # 3. Classical Selectors (Grid Search via Inner CV)
#     variance_grid = [{"threshold": t} for t in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]]
#     results.append(nested_cv_evaluation(X, y, variance_selector, variance_grid, "Variance", outer_splits=outer_splits))

#     anova_grid = [{"percentile": p} for p in [10, 25, 50, 75, 90]]
#     results.append(nested_cv_evaluation(X, y, anova_selector, anova_grid, "Anova", outer_splits=outer_splits))

#     mi_grid = [
#         {"percentile": p, "n_neighbors": k}
#         for p, k in itertools.product([10, 25, 50, 75, 90], [3, 5, 7])
#     ]
#     results.append(nested_cv_evaluation(X, y, mi_selector, mi_grid, "Mutual Information", outer_splits=outer_splits))

#     l1_grid = [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]
#     results.append(nested_cv_evaluation(X, y, l1logistic_selector, l1_grid, "L1 Logistic Regression", outer_splits=outer_splits))

#     rfe_grid = [
#         {"feature_ratio": r, "C": c}
#         for r, c in itertools.product([0.1, 0.25, 0.5, 0.75], [0.1, 1.0, 10.0])
#     ]
#     results.append(nested_cv_evaluation(X, y, rfe_selector, rfe_grid, "RFE (SVM)", outer_splits=outer_splits))

#     boruta_grid = [
#         {"perc": p, "max_iter": m}
#         for p, m in itertools.product([80, 90, 100], [50, 100])
#     ]
#     results.append(nested_cv_evaluation(X, y, boruta_selector, boruta_grid, "Boruta", outer_splits=outer_splits))

#     # 4. GFSIR Grid
#     gfsir_grid_params = [
#         {"gfsir_nfeatures": nf, "gfsir_minth": minth, "gfsir_maxth": maxth, "gfsir_selector": sel}
#         for nf, minth, maxth, sel in itertools.product(
#             config[dataset]['grid_params']['gfsir_nfeatures'],
#             config[dataset]['grid_params']['gfsir_minth'],
#             config[dataset]['grid_params']['gfsir_maxth'],
#             config[dataset]['grid_params']['gfsir_selector']
#         )
#     ]
#     results.append(nested_cv_evaluation(X, y, gfsir_grid, gfsir_grid_params, "GFSIR", outer_splits=outer_splits))

#     summary = pd.DataFrame(results)
#     summary = run_paired_wilcoxon_tests(summary)
#     summary.to_csv(f"outputs/{dataset}/{dataset}_benchmark_results.csv", index=False)

#     # --- PLOTTING PIPELINE ---
    
#     # 1. Main Unbiased Benchmark Comparisons (Outer Test CV)
#     plots.performance_boxplot(summary, dataset, metric="balanced_accuracy")
#     plots.feature_stability_plot(summary, dataset)
#     plots_pt.performance_boxplot_pt(summary, dataset, metric="balanced_accuracy")

#     # 2. DyGraFS Hyperparameter Sensitivity Plots (Inner CV Scores)
#     if dygrafs_inner_summary is not None and len(dygrafs_inner_summary) > 0:
#         # Create full evaluation data frame for sensitivity functions
#         grid_summary_full = pd.concat([summary, dygrafs_inner_summary], ignore_index=True)

#         plots.accuracy_vs_runtime_by_threshold(grid_summary_full, dataset)
#         plots.accuracy_vs_runtime_by_similarity_function(grid_summary_full, dataset)
#         plots.accuracy_vs_runtime_by_cn_selector(grid_summary_full, dataset)
#         plots.accuracy_vs_features_by_threshold(grid_summary_full, dataset)
#         plots.accuracy_vs_features_by_similarity_function(grid_summary_full, dataset)
#         plots.accuracy_vs_features_by_cn_selector(grid_summary_full, dataset)
#         plots.accuracy_vs_threshold_by_cn_selector(grid_summary_full, dataset)

#         # Portuguese Translations
#         # plots_pt.accuracy_vs_runtime_by_threshold_pt(grid_summary_full, dataset)
#         # plots_pt.accuracy_vs_runtime_by_similarity_function_pt(grid_summary_full, dataset)
#         # plots_pt.accuracy_vs_runtime_by_cn_selector_pt(grid_summary_full, dataset)
#         # plots_pt.accuracy_vs_features_by_threshold_pt(grid_summary_full, dataset)
#         # plots_pt.accuracy_vs_features_by_similarity_function_pt(grid_summary_full, dataset)
#         # plots_pt.accuracy_vs_features_by_cn_selector_pt(grid_summary_full, dataset)
#         # plots_pt.accuracy_vs_threshold_by_cn_selector_pt(grid_summary_full, dataset)

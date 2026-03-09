import pandas as pd
import time
import itertools
import json
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.svm import SVC
from pipeline.feature_selector import select_cn_centers
from sklearn.feature_selection import VarianceThreshold
import pipeline.model_plots as plots
from sklearn.model_selection import StratifiedKFold
import warnings
from itertools import combinations
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression
from pipeline.GFSIR.graph_feature_selection import GraphFeatureSelection
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(42)

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

# Feature stability calculated with Jaccard between folds
def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return np.nan
    return len(a & b) / len(a | b)

def compute_auroc(y_true, y_proba, model):
    """
    Computes AUROC for binary or multiclass datasets.
    Ensures consistency across CV folds by restricting to present classes.
    """
    present_classes = np.unique(y_true)

    # AUROC undefined if <2 classes
    if len(present_classes) < 2:
        return np.nan

    # Binary dataset
    if len(present_classes) == 2:
        pos_idx = list(model.classes_).index(present_classes.max())
        return roc_auc_score(y_true, y_proba[:, pos_idx])

    # Multiclass: restrict to present classes only
    mask = np.isin(model.classes_, present_classes)

    return roc_auc_score(
        y_true,
        y_proba[:, mask],
        labels=present_classes,
        multi_class="ovr",
        average="macro"
    )

# Bridge for calling selectors and running models between folds
def run_eval(model_data, selector_fn, selector_name, selector_params=None, kf=None):
    print("\nRunning with selector:", selector_name)
    if selector_params is not None:
        print(f"Params: {selector_params}")
    if kf is not None:
        return evaluate_with_kfold(
            kf,
            model_data['X'], model_data['y'],
            selector_fn=selector_fn,
            selector_name=selector_name,
            selector_params=selector_params
        )
    else:
        return evaluate_with_predefined_split(
            model_data['X_train'], model_data['X_test'], model_data['y_train'], model_data['y_test'],
            selector_fn=selector_fn,
            selector_name=selector_name,
            selector_params=selector_params
        )

# This module can be used if no kfold approach is available
def evaluate_with_predefined_split(
    X_train, X_test, y_train, y_test,
    selector_fn, selector_name, selector_params=None
):

    if selector_params is None:
        selector_params = {}
    else:
        selector_params = dict(selector_params)
    selector_params["seed"] = 42
    selector_params["save_fig"] = True

    start = time.time()

    if selector_fn is None:
        selected = X_train.columns.tolist()
    else:
        selected = selector_fn(X_train, y_train, selector_params)

    if len(selected) == 0:
        return None

    runtime = time.time() - start
    if selector_fn is None:
        runtime = 0

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train[selected], y_train)
    y_pred = model.predict(X_test[selected])
    y_proba = model.predict_proba(X_test[selected])

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auroc = compute_auroc(y_test, y_proba, model)

    print(f"acc: {acc}, bal_acc: {bal_acc}, auroc: {auroc}")
    print(f"Runtime: {runtime}, selected features: {len(selected)}")
    return {
        "selector": selector_name,
        "link_method": selector_params.get("link") if selector_params else None,
        "threshold": selector_params.get("threshold") if selector_params else None,
        "cn_selector": selector_params.get("cn_selector") if selector_params else None,
        "gfsir_nfeatures": selector_params.get("gfsir_nfeatures") if selector_params else None,
        "gfsir_minth": selector_params.get("gfsir_minth") if selector_params else None,
        "gfsir_maxth": selector_params.get("gfsir_maxth") if selector_params else None,
        "gfsir_selector": selector_params.get("gfsir_selector") if selector_params else None,
        "accuracy_mean": acc,
        "accuracy_std": 0.0,
        "balanced_accuracy_mean": bal_acc,
        "balanced_accuracy_std": 0.0,
        "auroc_mean": auroc,
        "auroc_std": 0.0,
        "runtime_mean": runtime,
        "features_mean": len(selected),
        "selected_features": selected
    }

# Select features, run the model and extract metrics
def evaluate_with_kfold(kf, X, y, selector_fn, selector_name, selector_params=None):

    # Copy the dictionary to avoid misuse out of the function
    if selector_params is None:
        selector_params = {}
    else:
        selector_params = dict(selector_params)  # defensive copy

    bal_accs = []
    accs = []
    runtimes = []
    selected_features_all = []
    n_features_all = []
    aurocs = []
    selector_params["save_fig"] = True

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()

        if selector_fn is None:
            selected = X.columns.tolist()
        else:
            selector_params["seed"] = 42 + fold
            selected = selector_fn(X_train, y_train, selector_params)
            selector_params["save_fig"] = False
            # Only the first fold will generate a plot

        # selected = [f for f in selected if f in X.columns]
        selected = list(set(selected).intersection(X.columns))
        if len(selected) == 0:
            warnings.warn(
                f"[Fold {fold}] No features selected by {selector_name}. Recording NaNs.",
                RuntimeWarning
            )
            accs.append(np.nan)
            bal_accs.append(np.nan)
            aurocs.append(np.nan)
            runtime = time.time() - start
            runtimes.append(runtime)
            selected_features_all.append([])
            n_features_all.append(0)
            continue

        runtime = time.time() - start

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42 + fold,
            class_weight="balanced"
        )
        model.fit(X_train[selected], y_train)
        y_pred = model.predict(X_test[selected])
        y_proba = model.predict_proba(X_test[selected])

        accs.append(accuracy_score(y_test, y_pred))
        bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        aurocs.append(compute_auroc(y_test, y_proba, model))
        if selector_fn is None:
            runtimes.append(0)
        else:
            runtimes.append(runtime)
        selected_features_all.append(selected)
        n_features_all.append(len(selected))

    if len(selected_features_all) > 1:
        stability = np.nanmean([
            jaccard(a, b)
            for a, b in combinations(selected_features_all, 2)
        ])
    else:
        stability = np.nan

    print(f"acc mean: {np.mean(accs)}, bal_acc mean: {np.mean(bal_accs)}, auroc mean: {np.mean(aurocs)}")
    print(f"Runtime mean: {np.mean(runtimes)}, selected features mean: {int(np.mean(n_features_all))}")
    return {
        "selector": selector_name,
        "link_method": selector_params.get("link") if selector_params else None,
        "threshold": selector_params.get("threshold") if selector_params else None,
        "cn_selector": selector_params.get("cn_selector") if selector_params else None,
        "gfsir_nfeatures": selector_params.get("gfsir_nfeatures") if selector_params else None,
        "gfsir_minth": selector_params.get("gfsir_minth") if selector_params else None,
        "gfsir_maxth": selector_params.get("gfsir_maxth") if selector_params else None,
        "gfsir_selector": selector_params.get("gfsir_selector") if selector_params else None,
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "balanced_accuracy_mean": np.mean(bal_accs),
        "balanced_accuracy_std": np.std(bal_accs),
        "auroc_mean": np.mean(aurocs),
        "auroc_std": np.std(aurocs),
        "feature_stability": stability,
        "runtime_mean": np.nanmean(runtimes),
        "features_mean": np.mean(n_features_all)
    }
    # "runtime_mean": np.mean(runtimes),

# included an L1-regularized logistic regression as a sparse linear baseline instead of LASSO
def l1logistic_selector(X_train, y_train, params=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_scaled, y_train)

    coef = np.abs(model.coef_).sum(axis=0)
    return X_train.columns[coef > 1e-6].tolist()

def mi_selector(X_train, y_train, params=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    scores = mutual_info_classif(X_scaled, y_train, random_state=42)
    threshold = np.median(scores)
    return X_train.columns[scores >= threshold].tolist()

def variance_selector(X_train, y_train, params=None):
    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X_train)
    return X_train.columns[vt.get_support()].tolist()

def rfe_selector(X_train, y_train, params=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    svc = SVC(kernel="linear", random_state=42)
    n_features = max(1, int(X_train.shape[1] * 0.5))

    rfe = RFE(
        estimator=svc,
        n_features_to_select=n_features
    )
    rfe.fit(X_scaled, y_train)

    return X_train.columns[rfe.support_].tolist()

def anova_selector(X_train, y_train, params=None):
    scores, _ = f_classif(X_train, y_train)
    threshold = np.nanmedian(scores)
    return X_train.columns[scores >= threshold].tolist()

def gfsir_grid(X_train, y_train, params=None):
    assert params is not None

    # You must obtain the GFSIR repository for requesting this selector
    selector = GraphFeatureSelection(
        input_dir=".",
        output_dir=".",
        lower_threshold=params["gfsir_minth"],
        upper_threshold=params["gfsir_maxth"],
        n_features=params["gfsir_nfeatures"]
    )

    # Automatic threshold definition
    if params["gfsir_minth"] == "auto":
        df_selected = selector.apply_graph_feature_selection(
            X_train.copy(),
            method=params["gfsir_selector"],
            mode="adaptive"
        )
        return df_selected.columns.tolist()

    # Threshold definition by providing bounds
    df_selected = selector.apply_graph_feature_selection(
        X_train.copy(),
        method=params["gfsir_selector"],
        mode="manual"
    )
    return df_selected.columns.tolist()

def boruta_selector(X_train, y_train, params=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Random Forest required by Boruta
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        random_state=42,
        verbose=0
    )

    y_array = y_train.values if hasattr(y_train, "values") else y_train
    boruta.fit(X_scaled, y_array)

    return X_train.columns[boruta.support_].tolist()

def graph_selector(X_train, y_train, params):
    assert params is not None
    image_filename = f"{params['dataset']}_{params['link']}_{params['threshold']:.2f}_{params['cn_selector']}_radiomic_graph.png"
    return select_cn_centers(
        X_train,
        threshold=params["threshold"],
        cn_selector=params["cn_selector"],
        link_method=params["link"],
        seed_nb=params["seed"],
        save_fig=params["save_fig"],
        png_path = f"outputs/{params['dataset']}/feature_plots/{image_filename}",
    )

# Estimative of the best threshold values for a given dataset
# currently only checking Pearson and Spearman
def estimate_best_graph_params(X):
    """
    Unsupervised estimator of graph parameters.
    Uses only feature correlations.
    """

    results = {}
    print("\n=== Automatic Graph Parameter Estimation ===")

    link_methods = ("pearson", "spearman")

    # Find the threshold interval where only 15% of the data survives
    # this range will have a low amount of correlated features, and will
    # have a smaller cost comparing with the complete network
    TARGET_DENSITY = 0.15
    TH_GRID = np.linspace(0.3, 0.9, 61)

    for link in link_methods:
        corr = X.corr(method=link)

        # Upper triangle only
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        vals = np.abs(upper.values)
        vals = vals[~np.isnan(vals)]
        vals = vals[vals < 0.99]  # drop near-duplicate features

        mean_corr = vals.mean()
        median_corr = np.median(vals)

        densities = np.array([np.mean(vals >= th) for th in TH_GRID])
        best_idx = np.argmin(np.abs(densities - TARGET_DENSITY))
        # Forcing networking density to reach 0.15

        base_th = TH_GRID[best_idx]
        density = densities[best_idx]
        th_range = (max(0.0, base_th - 0.1), min(1.0, base_th + 0.1))

        results[link] = {
            "mean_corr": mean_corr,
            "median_corr": median_corr,
            "threshold": base_th,
            "density": density,
            "range": th_range
        }

    # Select best link method (balanced density)
    best_link = min(
        results.keys(),
        key=lambda k: abs(results[k]["density"] - TARGET_DENSITY)
    )

    best = results[best_link]

    print("\n=== Recommended Parameters ===")
    print(f"link_method   : {best_link}")
    print(f"threshold     : {best['threshold']:.2f}")
    print(f"density       : {best['density']:.3f}")
    print(f"threshold_rng : {best['range']}")

    return f"Expected ideal threshold range: {best['threshold']}"

def run_model_with_splits(X_train, X_test, y_train, y_test, description=""):
    # The baseline model is always Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42,class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n=== Results {description} ===")
    print(f"Accuracy: {acc:.4f}, Bal acc: {bal_acc:.4f}")
    return acc, bal_acc

def model_benchmarking(dataset="sample"):

    thresholds = config[dataset]['grid_params']['thresholds']
    link_methods = config[dataset]['grid_params']['link_methods']
    cn_selectors = config[dataset]['grid_params']['cn_selectors']
    gfsir_nfeatures = config[dataset]['grid_params']['gfsir_nfeatures']
    gfsir_minth = config[dataset]['grid_params']['gfsir_minth']
    gfsir_maxth = config[dataset]['grid_params']['gfsir_maxth']
    gfsir_selector = config[dataset]['grid_params']['gfsir_selector']

    # Loading the original radiomic features
    radiomic_features_path = f"{config[dataset]['output_path']}{dataset}_radiomic_features.csv"
    df = pd.read_csv(radiomic_features_path)

    total_time_start = time.time()

    # Drop columns not related to the features
    tg_column = config[dataset]["target_column"]
    model_data = {}

    # Remove rows where target (y) is NaN
    df = df.dropna(subset=[tg_column])

    # The NSCLC dataset has a class with only 2 obvervations
    # A version with all dataset will be used, and also one dropping this class
    # This was a blocker to a wider CV size
    if "drop_rare_classes" in dataset:
        class_counts = df[tg_column].value_counts()
        valid_classes = class_counts[class_counts >= 5].index

        removed = set(class_counts.index) - set(valid_classes)
        if len(removed) > 0:
            print(f"Dropping rare classes: {removed}")

        df = df[df[tg_column].isin(valid_classes)]

    static_remove = [tg_column, "exam_path", "gt_path", "patient_id"]
    dynamic_remove = config[dataset].get("to_remove_columns", [])
    columns_to_remove = static_remove + dynamic_remove
    model_data['X'] = df.drop(columns=columns_to_remove, errors="ignore")
    
    le = LabelEncoder() # Converting target from str to nmb (required for NSCLC)
    y = le.fit_transform(df[tg_column])
    model_data['y'] = np.asarray(y)
    use_kfold = True

    # Storing results to final benchmarking
    results = []

    kf = None
    if use_kfold:
        # Ensure each CV fold contains all classes (prevents invalid AUROC computation)
        min_class_size = np.min(np.bincount(y))
        n_splits = min(5, min_class_size)

        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        print(f"Number of folds used: {n_splits}")
        estimate_best_graph_params(model_data['X'])
    else:
        estimate_best_graph_params(model_data['X_train'])

    # GFSIR selector, please, extract it from the author
    # https://github.com/hmMed22/GFSIR
    # results.append(run_eval(model_data, gfsir_connected, "GFSIR Connected", kf=kf))
    # results.append(run_eval(model_data, gfsir_louvain, "GFSIR Louvain", kf=kf))
    # results.append(run_eval(model_data, gfsir_spectral, "GFSIR Spectral", kf=kf))
    for nfeatures, minth, maxth, selector in itertools.product(gfsir_nfeatures, gfsir_minth, gfsir_maxth, gfsir_selector):
   
        params = {
            "gfsir_nfeatures": nfeatures,
            "gfsir_minth": minth,
            "gfsir_maxth": maxth,
            "gfsir_selector": selector,
        }

        results.append(run_eval(model_data, gfsir_grid, "GFSIR", selector_params=params, kf=kf))

    # Classical Feature Selectors from literature
    print("\nRunning classical feature selectors...")
    results.append(run_eval(model_data,  None, "Vanilla RF", kf=kf))
    results.append(run_eval(model_data, variance_selector, "Variance", kf=kf))
    results.append(run_eval(model_data, anova_selector, "Anova", kf=kf))
    results.append(run_eval(model_data, mi_selector, "Mutual Information", kf=kf))
    results.append(run_eval(model_data, l1logistic_selector, "L1 Logistic Regression", kf=kf))
    results.append(run_eval(model_data, rfe_selector, "RFE (SVM)", kf=kf))
    results.append(run_eval(model_data, boruta_selector, "Boruta", kf=kf))

    # Checking complex network feature selector with multiple parameters
    for link, th, cn, in itertools.product(link_methods, thresholds, cn_selectors):

        params = {
            "dataset": dataset,
            "threshold": th,
            "cn_selector": cn,
            "link": link,
        }

        results.append(run_eval(model_data, graph_selector, "DyGraFS", selector_params=params, kf=kf))

    summary = pd.DataFrame(results)
    summary.to_csv(f"outputs/{dataset}/{dataset}_benchmark_results.csv", index=False)
    outfile = f"outputs/{dataset}/results_metadata.txt"

    with open(outfile, "a") as f:

        elapsed = time.time() - total_time_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        f.write(f"Total runtime: {hours}h {minutes}m {seconds}s\n")
        if use_kfold:
            f.write(f"Total samples: {model_data['X'].shape[0]}\n")
        else:
            f.write(f"Total samples: {model_data['X_train'].shape[0]}\n")

        if kf is not None:
            f.write(f"CV strategy: {kf.get_n_splits()}-fold StratifiedKFold\n")

        print(f"\nResults saved to {dataset}_benchmark_results.csv\n")

    plots.print_cn_performance_summary(outfile, summary)

    df_plot = summary[
        (summary["selector"] == "DyGraFS") &
        (summary["threshold"].notna())
    ].copy()

    if len(df_plot) > 0:
        plots.accuracy_vs_runtime_by_threshold(summary, dataset)
        plots.accuracy_vs_runtime_by_link_method(summary, dataset)
        plots.accuracy_vs_runtime_by_cn_selector(summary, dataset)
        plots.accuracy_vs_features_by_threshold(summary, dataset)
        plots.accuracy_vs_features_by_link_method(summary, dataset)
        plots.accuracy_vs_features_by_cn_selector(summary, dataset)
        plots.performance_boxplot(summary, dataset, metric="balanced_accuracy")
import pandas as pd
import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pipeline.feature_selector import select_community_centers
import json
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.svm import SVC

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

def run_model(X, y, description=""):
    """Train and evaluate a Random Forest model."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== Results {description} ===")
    print(f"Accuracy: {acc:.4f}")
    return acc


def model_benchmarking(dataset="brats_africa"):
    radiomic_features_path = f"{config[dataset]["output_folder"]}{dataset}_radiomic_features.csv"
    df = pd.read_csv(radiomic_features_path)
    total_time_start = time.time()

    # Drop columns not related to the features
    X = df.drop(columns=["glioma", "exam_path", "gt_path", "patient_id"])
    y = df["glioma"] # Target definition

    # Parameter grids
    thresholds = [0.3, 0.45, 0.6, 0.7, 0.8]
    link_methods = ["cosine", "spearman", "pearson", "rho_distance"]
    community_methods = ["lp", "pr"]
    eigen_options = [False, True]
    classical_selectors = ["lasso", "information_gain", "gini", "rfe"]

    # Store results
    results = []

    # Control model
    print("\nRunning control model (all features)")
    # t0 = time.time()
    acc_all = run_model(X, y, description="(All features)")
    # control_time = time.time() - t0
    results.append({
        "Selector": "none"
        "link_method": "none",
        "threshold": None,
        "community_method": "none",
        "check_eigen": None,
        "accuracy": acc_all,
        "runtime(sec)": "none",
        "features nb": len(X.columns)
    })

    # Classical Feature Selectors (Lasso, InfoGain, Gini, RFE)
    print("\nRunning classical feature selectors...")

    for selector in classical_selectors:
        print(f"\n--- Running {selector} selector ---")
        start_time = time.time()

        selected = []
        try:
            if selector == "lasso":
                model = LassoCV(cv=5, random_state=42).fit(X, y)
                selected = X.columns[model.coef_ != 0]

            elif selector == "information_gain":
                scores = mutual_info_classif(X, y, random_state=42)
                threshold = np.median(scores)  # keep top 50%
                selected = X.columns[scores >= threshold]

            elif selector == "gini":
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(X, y)
                importances = rf.feature_importances_
                threshold = np.median(importances)
                selected = X.columns[importances >= threshold]

            elif selector == "rfe":
                svc = SVC(kernel="linear", random_state=42)
                rfe = RFE(svc, n_features_to_select=int(len(X.columns) * 0.5))
                rfe.fit(X, y)
                selected = X.columns[rfe.support_]

            if len(selected) == 0:
                print(f"No features selected by {selector}; skipping.")
                continue

            runtime = time.time() - start_time
            acc = run_model(X[selected], y, description=f"({selector})")

            results.append({
                "selector": selector,
                "link_method": "none",
                "threshold": None,
                "community_method": "none",
                "check_eigen": None,
                "accuracy": acc,
                "runtime(sec)": runtime,
                "features nb": len(selected)
            })
        except Exception as e:
            print(f"Error with {selector}: {e}")
            continue

    # == GRID SEARCH ==
    # Checking complex network feature selector with multiple parameters
    print("\nRunning feature-selection models(gr)")
    for link, th, cm, eigen in itertools.product(link_methods, thresholds, community_methods, eigen_options):
        desc = f"({link}, thr={th}, {cm}, eigen={eigen})"
        print(f"\nRunning {desc}")
        start_time = time.time()

        selected = select_community_centers(
            radiomic_features_path,
            threshold=th,
            mapping_file=f"mapping_{link}_{cm}.csv",
            png_file=f"graph_{link}_{cm}.png",
            method=cm,
            check_eigen=eigen,
        )

        # Keep intersection of selected features
        selected = [f for f in selected if f in X.columns]
        if len(selected) == 0:
            print("No features selected; skipping.")
            continue

        runtime = time.time() - start_time
        acc = run_model(X[selected], y, description=desc)

        results.append({
            "selector": "complex network"
            "link_method": link,
            "threshold": th,
            "community_method": cm,
            "check_eigen": eigen,
            "accuracy": acc,
            "runtime(sec)": runtime,
            "features nb": len(selected)
        })

    print("\n==============================")
    print("FINAL BENCHMARK SUMMARY")
    print("==============================")
    summary = pd.DataFrame(results)
    print(summary.sort_values(by="accuracy", ascending=False).reset_index(drop=True))

    elapsed = time.time() - total_start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Total runtime: {hours}h {minutes}m {seconds}s")

    # Saving results
    summary.to_csv(f"{dataset}_benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

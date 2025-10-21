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

import networkx as nx
import matplotlib.pyplot as plt


def analyze_and_prepare_features(csv_path, save_clean=True, plot_corr_graph=False, corr_threshold=0.0):
    """
    Analyze and prepare radiomic features before model benchmarking.

    Parameters
    ----------
    csv_path : str
        Path to the radiomic features CSV file.
    save_clean : bool
        Whether to save the cleaned and normalized CSV file.
    plot_corr_graph : bool
        Whether to plot a feature correlation network.
    corr_threshold : float
        Minimum correlation threshold to define edges in the correlation graph.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned and normalized DataFrame ready for model training.
    """

    print(f"\n🔍 Loading and analyzing radiomic features from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"\n--- DATA OVERVIEW ---")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # === 1️⃣ Check for nulls ===
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\n⚠️ Missing values detected:")
        print(null_counts[null_counts > 0])
        df = df.dropna()
        print(f"➡️ Dropped rows with nulls. New shape: {df.shape}")
    else:
        print("\n✅ No missing values detected.")

    # === 2️⃣ Drop non-numeric columns (if any) ===
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"\nℹ️ Dropping non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    # === 3️⃣ Detect constant columns ===
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\n⚠️ Constant columns detected: {constant_cols}")
        df = df.drop(columns=constant_cols)
        print(f"➡️ Dropped {len(constant_cols)} constant columns.")
    else:
        print("\n✅ No constant columns detected.")

    # === 4️⃣ Check if already normalized ===
    means = df.mean()
    stds = df.std()
    if np.allclose(means, 0, atol=0.1) and np.allclose(stds, 1, atol=0.1):
        print("\n✅ Data appears to be already normalized (mean≈0, std≈1).")
        df_clean = df.copy()
    else:
        print("\n⚙️ Applying z-score normalization...")
        scaler = StandardScaler()
        df_clean = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        print("✅ Normalization complete.")

    # === 5️⃣ Optional: correlation graph check ===
    if plot_corr_graph:
        print("\n📊 Building correlation network...")
        corr = df_clean.corr()
        G = nx.Graph()
        for col in corr.columns:
            G.add_node(col)
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= corr_threshold:
                    G.add_edge(corr.columns[i], corr.columns[j])
        n_components = nx.number_connected_components(G)
        print(f"🔗 Correlation graph has {n_components} connected component(s).")
        if n_components > 1:
            print("⚠️ The feature graph is not fully connected — "
                  "some features are weakly correlated.")
        if plot_corr_graph:
            plt.figure(figsize=(10, 8))
            nx.draw(G, with_labels=True, node_size=600, font_size=8,
                    node_color='skyblue', edge_color='gray')
            plt.title("Radiomic Feature Correlation Graph")
            plt.show()

    # === 6️⃣ Save cleaned data ===
    if save_clean:
        clean_path = csv_path.replace(".csv", "_cleaned.csv")
        df_clean.to_csv(clean_path, index=False)
        print(f"\n💾 Cleaned and normalized file saved to: {clean_path}")

    print("\n✅ Feature analysis and preparation complete.\n")
    return df_clean


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

    # Parameter grids
    # thresholds = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8]
    thresholds = [0.0, 0.15, 0.3, 0.45, 0.7]
    link_methods = ["cosine", "spearman", "pearson", "rho_distance"]
    community_methods = ["lp", "pr"]
    eigen_options = [False, True]
    classical_selectors = ["lasso", "information_gain", "gini"]

    # rfe takes forever
    # thresholds = [0.0, 0.3]
    # link_methods = ["pearson"]
    # community_methods = ["lp", "pr"]
    # eigen_options = [False]
    # classical_selectors = ["lasso", "information_gain", "gini"]

    radiomic_features_path = f"{config[dataset]['output_path']}{dataset}_radiomic_features.csv"

    # check_df = analyze_and_prepare_features(
    #     radiomic_features_path,
    #     save_clean=True,
    #     plot_corr_graph=True
    # )

    df = pd.read_csv(radiomic_features_path)
    total_time_start = time.time()

    # Drop columns not related to the features
    X = df.drop(columns=["glioma", "exam_path", "gt_path", "patient_id"])
    y = df["glioma"] # Target definition

    # Store results
    results = []

    # Control model
    print("\nRunning control model (all features)")
    # t0 = time.time()
    acc_all = run_model(X, y, description="(All features)")
    # control_time = time.time() - t0
    results.append({
        "selector": "none",
        "link_method": "none",
        "threshold": None,
        "community_method": "none",
        "check_eigen": None,
        "accuracy": acc_all,
        "runtime(sec)": "none",
        "features nb": len(X.columns),
        "selected features": ["all"],
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
                "features nb": len(selected),
                "selected features": selected,
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
            link_method=link,
        )

        # Keep intersection of selected features
        selected = [f for f in selected if f in X.columns]
        if len(selected) == 0:
            print("No features selected; skipping.")
            continue

        runtime = time.time() - start_time
        acc = run_model(X[selected], y, description=desc)

        results.append({
            "selector": "complex network",
            "link_method": link,
            "threshold": th,
            "community_method": cm,
            "check_eigen": eigen,
            "accuracy": acc,
            "runtime(sec)": runtime,
            "features nb": len(selected),
            "selected features": selected,
        })

    print("\n==============================")
    print("FINAL BENCHMARK SUMMARY")
    print("==============================")
    summary = pd.DataFrame(results)
    print(summary.sort_values(by="accuracy", ascending=False).reset_index(drop=True))

    elapsed = time.time() - total_time_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Total runtime: {hours}h {minutes}m {seconds}s")

    # Saving results
    summary.to_csv(f"{dataset}_benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

import pandas as pd
import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pipeline.select_features import select_community_centers

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


def model_benchmarking():
    df = pd.read_csv("outputs/brats_africa/radiomic_features_brats_africa.csv")

    X = df.drop(columns=["glioma", "exam_path", "gt_path", "patient_id"])
    y = df["glioma"]

    # --- Parameter grids ---
    thresholds = [0.6, 0.7, 0.8]
    link_methods = ["cosine", "spearman", "pearson"]
    community_methods = ["lp", "pr"]
    eigen_options = [False, True]

    # Store results
    results = []

    # --- Control model ---
    print("\nRunning control model (all features)")
    t0 = time.time()
    acc_all = run_model(X, y, description="(All features)")
    control_time = time.time() - t0
    results.append({
        "link_method": "none",
        "threshold": None,
        "community_method": "none",
        "check_eigen": None,
        "accuracy": acc_all,
        "runtime": control_time
    })

    # --- Loop over combinations ---
    print("\nRunning feature-selection models(gr)")
    for link, th, cm, eigen in itertools.product(link_methods, thresholds, community_methods, eigen_options):
        desc = f"({link}, thr={th}, {cm}, eigen={eigen})"
        print(f"\n--- Running {desc} ---")
        start_time = time.time()

        selected = select_community_centers(
            "outputs/brats_africa/radiomic_features_brats_africa.csv",
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

        acc = run_model(X[selected], y, description=desc)
        runtime = time.time() - start_time

        results.append({
            "link_method": link,
            "threshold": th,
            "community_method": cm,
            "check_eigen": eigen,
            "accuracy": acc,
            "runtime": runtime
        })

    # --- Summary table ---
    print("\n==============================")
    print("FINAL BENCHMARK SUMMARY")
    print("==============================")
    summary = pd.DataFrame(results)
    print(summary.sort_values(by="accuracy", ascending=False).reset_index(drop=True))
    summary.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")


# import pandas as pd
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from pipeline.select_features import select_community_centers


# def run_model(X, y, description=""):
#     """Train and evaluate a Random Forest model."""
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y.astype(str))

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = RandomForestClassifier(n_estimators=200, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)

#     print(f"\n=== Results {description} ===")
#     print(f"Accuracy: {acc:.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=le.classes_))
#     return acc


# def model_benchmarking():
#     # Load data
#     df = pd.read_csv("outputs/brats_africa/radiomic_features_brats_africa.csv")

#     # Separate features and target
#     X = df.drop(columns=["glioma", "exam_path", "gt_path", "patient_id"])
#     y = df["glioma"]

#     results = {}

#     # --- Round 1: Control model (all features) ---
#     print("\n🔹 Running control model (all features)...")
#     start_time = time.time()
#     acc_all = run_model(X, y, description="(All features)")
#     results["All features"] = (acc_all, time.time() - start_time)

#     # --- Round 2: Label Propagation (LP) ---
#     print("\n🔹 Running model with Label Propagation feature selection...")
#     start_time = time.time()
#     selected_lp = select_community_centers(
#         "outputs/brats_africa/radiomic_features_brats_africa.csv",
#         threshold=0.7,
#         mapping_file="brats_africa_feature_mapping.csv",
#         png_file="brats_africa_radiomic_graph.png",
#         method="lp",
#         check_eigen=False
#     )
#     X_lp = X[selected_lp]
#     acc_lp = run_model(X_lp, y, description="(Label Propagation)")
#     results["Label Propagation"] = (acc_lp, time.time() - start_time)

#     # --- Round 3: PageRank (PR) ---
#     print("\n🔹 Running model with PageRank feature selection...")
#     start_time = time.time()
#     selected_pr = select_community_centers(
#         "outputs/brats_africa/radiomic_features_brats_africa.csv",
#         threshold=0.7,
#         mapping_file="brats_africa_feature_mapping.csv",
#         png_file="brats_africa_radiomic_graph.png",
#         method="pr",
#         check_eigen=False
#     )
#     X_pr = X[selected_pr]
#     acc_pr = run_model(X_pr, y, description="(PageRank)")
#     results["PageRank"] = (acc_pr, time.time() - start_time)

#     # --- Round 4: Label Propagation + Eigenvalue ---
#     print("\n🔹 Running model with Label Propagation + Eigenvalue refinement...")
#     start_time = time.time()
#     selected_lp_eigen = select_community_centers(
#         "outputs/brats_africa/radiomic_features_brats_africa.csv",
#         threshold=0.7,
#         mapping_file="brats_africa_feature_mapping.csv",
#         png_file="brats_africa_radiomic_graph.png",
#         method="lp",
#         check_eigen=True
#     )
#     X_lp_eigen = X[selected_lp_eigen]
#     acc_lp_eigen = run_model(X_lp_eigen, y, description="(Label Propagation + Eigenvalue)")
#     results["Label Propagation + Eigen"] = (acc_lp_eigen, time.time() - start_time)

#     # --- Round 5: PageRank + Eigenvalue ---
#     print("\n🔹 Running model with PageRank + Eigenvalue refinement...")
#     start_time = time.time()
#     selected_pr_eigen = select_community_centers(
#         "outputs/brats_africa/radiomic_features_brats_africa.csv",
#         threshold=0.7,
#         mapping_file="brats_africa_feature_mapping.csv",
#         png_file="brats_africa_radiomic_graph.png",
#         method="pr",
#         check_eigen=True
#     )
#     X_pr_eigen = X[selected_pr_eigen]
#     acc_pr_eigen = run_model(X_pr_eigen, y, description="(PageRank + Eigenvalue)")
#     results["PageRank + Eigen"] = (acc_pr_eigen, time.time() - start_time)

#     # --- Summary ---
#     print("\n==============================")
#     print("🏁 FINAL BENCHMARK SUMMARY")
#     print("==============================")
#     for label, (acc, t) in results.items():
#         print(f"{label:<30}  Accuracy: {acc:.4f} | Time: {t:.2f} seconds")


# if __name__ == "__main__":
#     model_benchmarking()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from pipeline.select_features import select_community_centers

# def run_model(X, y, description=""):
#     """Train and evaluate a Random Forest model."""
#     le = LabelEncoder()
#     # y_encoded = le.fit_transform(y)
#     y_encoded = le.fit_transform(y.astype(str))

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = RandomForestClassifier(n_estimators=200, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)

#     print(f"\n=== Results {description} ===")
#     print(f"Accuracy: {acc:.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=le.classes_))
#     return acc


# def model_benchmarking():
#     # Load data
#     df = pd.read_csv("outputs/brats_africa/radiomic_features_brats_africa.csv")

#     # Separate features and target
#     X = df.drop(columns=["glioma", "exam_path", "gt_path", "patient_id"])
#     y = df["glioma"]

#     # --- Round 1: All features ---
#     acc_all = run_model(X, y, description="(All features)")

#     # --- Round 2: Selected features only ---
#     selected_features = select_community_centers("outputs/brats_africa/radiomic_features_brats_africa.csv", threshold=0.7, mapping_file="brats_africa_feature_mapping.csv", png_file="brats_africa_radiomic_graph.png")
#     print(f"\nSelected {len(selected_features)} community center features.")
#     print(selected_features)

#     # Keep only selected features that exist in X
#     X_selected = X[selected_features]
#     acc_selected = run_model(X_selected, y, description="(Selected features)")

#     print("\n=== Summary ===")
#     print(f"Accuracy (All features): {acc_all:.4f}")
#     print(f"Accuracy (Selected features): {acc_selected:.4f}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pipeline.select_features import select_community_centers

def run_model(X, y, description=""):
    """Train and evaluate a Random Forest model."""
    le = LabelEncoder()
    # y_encoded = le.fit_transform(y)
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
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return acc


def model_benchmarking():
    # Load data
    df = pd.read_csv("outputs/brats_africa/radiomic_features_brats_africa.csv")

    # Separate features and target
    X = df.drop(columns=["glioma", "exam_path", "gt_path", "patient_id"])
    y = df["glioma"]

    # --- Round 1: All features ---
    acc_all = run_model(X, y, description="(All features)")

    # --- Round 2: Selected features only ---
    selected_features = select_community_centers("outputs/brats_africa/radiomic_features_brats_africa.csv", threshold=0.7, mapping_file="brats_africa_feature_mapping.csv", png_file="brats_africa_radiomic_graph.png")
    print(f"\nSelected {len(selected_features)} community center features.")
    print(selected_features)

    # Keep only selected features that exist in X
    X_selected = X[selected_features]
    acc_selected = run_model(X_selected, y, description="(Selected features)")

    print("\n=== Summary ===")
    print(f"Accuracy (All features): {acc_all:.4f}")
    print(f"Accuracy (Selected features): {acc_selected:.4f}")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# def main():
#     # Load features
#     df = pd.read_csv("outputs/brats_africa/radiomic_features_brats_africa.csv")

#     # --- 1️⃣ Remove non-feature columns and duplicates ---
#     # Keep patient_id separately for splitting
#     df = df.drop_duplicates()

#     # Separate out identifiers and target
#     patient_ids = df['patient_id']
#     y = df['glioma']
#     X = df.drop(columns=['glioma', 'exam_path', 'gt_path', 'patient_id'])

#     # Encode target
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)

#     # --- 2️⃣ Split based on unique patient IDs ---
#     unique_ids = df['patient_id'].unique()
#     # train_ids, test_ids = train_test_split(
#     #     unique_ids, test_size=0.2, random_state=42, 
#     #     stratify=df.drop_duplicates('patient_id')['glioma']
#     # )
#     patient_labels = df.groupby('patient_id')['glioma'].first()  # One label per patient
#     train_ids, test_ids = train_test_split(
#         patient_labels.index, test_size=0.2, random_state=42,
#         stratify=patient_labels
#     )

#     # Mask rows by patient ID
#     train_mask = patient_ids.isin(train_ids)
#     test_mask = patient_ids.isin(test_ids)

#     X_train, X_test = X[train_mask], X[test_mask]
#     y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]

#     # --- 3️⃣ Scale features ---
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # --- 4️⃣ Train classifier ---
#     model = RandomForestClassifier(n_estimators=200, random_state=42)
#     model.fit(X_train, y_train)

#     # --- 5️⃣ Evaluate ---
#     y_pred = model.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

# if __name__ == "__main__":
#     main()

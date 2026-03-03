import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from feature_selector import select_cn_centers
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)

with open('input/config.json', 'r') as file:
    config = json.load(file)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def run_model(X_train, y_train, X_test, patient_ids, selector_fn, selector_params):

    selected = selector_fn(X_train, y_train, selector_params)
    selected = [f for f in selected if f in X_train.columns]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Cross-validation accuracy (5-fold)
    cv_scores = cross_val_score(
        model,
        X_train[selected],
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print("Cross-validation accuracy:", cv_scores.mean())
    print("CV std:", cv_scores.std())

    # Train on full training data
    model.fit(X_train[selected], y_train)

    # Training accuracy (for reference only)
    train_pred = model.predict(X_train[selected])
    train_acc = accuracy_score(y_train, train_pred)
    print("Training accuracy:", train_acc)

    print("Selected features:", len(selected))
    print(selected)

    y_pred = model.predict(X_test[selected])

    submission = pd.DataFrame({
        "patientID": patient_ids,
        "Mutacion": y_pred.astype(int)
    })

    submission.to_csv("kaggle_prediction.csv", index=False)

    print("kaggle_prediction.csv generated.")


def graph_selector(X_train, y_train, params):
    assert params is not None
    return select_cn_centers(
        X_train,
        threshold=params["threshold"],
        cn_selector=params["cn_selector"],
        link_method=params["link"],
        seed_nb=params["seed"],
        save_fig=False,
        png_path = f"outputs/{params['dataset']}/feature_plots/{params['dataset']}_{params['link']}_{params['threshold']:.2f}_{params['cn_selector']}_radiomic_graph.png",
    )

def run_complex_network_selector(dataset="radiomics_lgg"):

    radiomic_features_path = f"{config[dataset]['output_path']}{dataset}_radiomic_features.csv"
    df = pd.read_csv(radiomic_features_path)

    test_path = f"{config[dataset]['output_path']}test.csv"
    df_test = pd.read_csv(test_path)

    tg_column = config[dataset]["target_column"]

    # Remove rows where target is NaN
    df = df.dropna(subset=[tg_column])

    columns_to_remove = ["Mutacion", "patientID"]

    X_train = df.drop(columns=columns_to_remove, errors="ignore")
    X_test = df_test.drop(columns=columns_to_remove, errors="ignore")

    le = LabelEncoder()
    y_train = le.fit_transform(df[tg_column])

    params = {
        "threshold": 0.7,
        "cn_selector": "Label Propagation",
        "link": "Cosine",
        "seed": 42,
        "dataset": dataset
    }

    run_model(
        X_train,
        y_train,
        X_test,
        df_test["patientID"],
        selector_fn=graph_selector,
        selector_params=params
    )

run_complex_network_selector(dataset="radiomics_lgg")
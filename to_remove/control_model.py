# loads data from the file upenn_paths.csv
# check columns exam_path and gt_path
# extracts radiomic features based on the image based on the ground truth as the mask(the images are in nii.gz)
# there are around 270 images, the model must extract all radiomic features into a final csv dataframe, and then
# train with 80/20 for a given target at the original csv called MGMT,
# that contains the values Methylated and Unmethylated

# the code is sepparated in two files: one loads all radiomic features into the final csv, the 
# other trains the model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load extracted features
    df = pd.read_csv("radiomic_features.csv")

    # Drop non-feature columns
    X = df.drop(columns=["MGMT", "exam_path", "gt_path"])
    y = df["MGMT"]

    # Encode target (Methylated / Unmethylated)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train classifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()

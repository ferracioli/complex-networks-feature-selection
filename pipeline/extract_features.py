import os
import json
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

# Remotion_list is a variable used to continue the mapping if it gets interrupted
def generate_exam_dataframe(max_items=None, remotion_list=[], dataset="brats_africa"):
    glioma_path = config[dataset]["glioma_path"]
    others_path = config[dataset]["others_path"]
    clinical_csv = config[dataset]["clinical_csv"]
    output_csv = f"{config[dataset]['output_path']}{dataset}_paths.csv"

    # Loading the clinical CSV
    clinical_df = pd.read_csv(clinical_csv, sep=",")
    print(clinical_df.columns)
    # Mapping the IDs according to the column name
    if "ID" in clinical_df.columns:
        id_col = "ID"
    else:
        id_col = clinical_df.columns[0]

    # Start mapping the paths from the dataset
    exam_list = []

    # Glioma images
    if os.path.exists(glioma_path):
        exam_folders = os.listdir(glioma_path)

        for exam in exam_folders:
            if max_items and len(exam_list) >= max_items:
                break

            if exam in remotion_list:
                print(f"{exam} already verified...")
                continue

            # Original exam
            exam_file = os.path.join(glioma_path, exam, f"{exam}-t2f.nii.gz")
            if not os.path.isfile(exam_file):
                continue

            # Manual segmentation(ground truth)
            gt_file = os.path.join(glioma_path, exam, f"{exam}-seg.nii.gz")

            exam_list.append({
                "exam_id": exam,
                "exam_path": exam_file,
                "gt_path": gt_file
            })

    # Non Glioma images
    if os.path.exists(others_path):
        exam_folders = os.listdir(others_path)
        print(exam_folders)

        for exam in exam_folders:
            if max_items and len(exam_list) >= max_items:
                break

            if exam in remotion_list:
                print(f"{exam} already verified...")
                continue

            # Original exam
            exam_file = os.path.join(others_path, exam, f"{exam}-t2f.nii.gz")
            if not os.path.isfile(exam_file):
                continue

            # Manual segmentation(ground truth)
            gt_file = os.path.join(others_path, exam, f"{exam}-seg.nii.gz")

            exam_list.append({
                "exam_id": exam,
                "exam_path": exam_file,
                "gt_path": gt_file
            })

    # Converting the list to a Pandas dataframe
    exams_df = pd.DataFrame(exam_list)

    # Inner join between the paths and clinical informations
    merged_df = clinical_df.merge(exams_df, left_on=id_col, right_on="exam_id", how="inner")

    # Storing the dataframe
    merged_df.to_csv(output_csv, index=False)

    print(f"Dataframe stored as: {output_csv}, size: {merged_df.shape}")
    return merged_df


def extract_radiomic_features(
    bin_width=40,
    normalize=False,
    min_roi_size=100,
    min_roi_dim=2,
    dataset="brats_africa",
):
    """
    Extract all available PyRadiomics features from MRI images.
    By default, loads CSV at "outputs/brats_africa/brats_africa_paths.csv"
    and saves results to "outputs/brats_africa/radiomic_features_brats_africa.csv".
    
    Args:
        bin_width (int): Bin width for intensity discretization.
        normalize (bool): Whether to normalize images before extraction.
        min_roi_size (int): Minimum ROI size.
        min_roi_dim (int): Minimum ROI dimension (2D or 3D).
        dataset (str): selected dataset to identify in the json config
        
    Returns:
        pd.DataFrame: Extracted radiomic features.
    """

    input_csv = f"{config[dataset]['output_path']}{dataset}_paths.csv"
    output_csv = f"{config[dataset]['output_path']}{dataset}_radiomic_features.csv"

    # Loading the csv containing exam paths
    df = pd.read_csv(input_csv)

    # Configuring extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings.update({
        "binWidth": bin_width,
        "normalize": normalize,
        "normalizeScale": 1,
        "minimumROIDimensions": min_roi_dim,
        "minimumROISize": min_roi_size,
        "enableMetadata": False
    })

    # Enable all feature classes
    extractor.disableAllFeatures()
    extractor.enableAllFeatures()

    # Enable all image types (original, wavelet, LoG, etc.)
    extractor.enableAllImageTypes()

    print("Enabled all PyRadiomics features and image types.")

    # Feature extraction
    all_features = []

    # For each exam
    for idx, row in df.iterrows():
        # Identify the paths to the exam and ground truth
        img_path = row["exam_path"]
        mask_path = row["gt_path"]
        glioma_target = row["glioma"]
        patient_id = row["ID"]

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"[{idx}] Skipping missing file: {img_path} or {mask_path}")
            continue

        try:
            # Reading images
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)

            features = extractor.execute(image, mask)

            # Keep only radiomic feature values (exclude diagnostics)
            clean_features = {k: v for k, v in features.items() if not k.startswith("diagnostics")}
            clean_features["glioma"] = glioma_target
            clean_features["exam_path"] = img_path
            clean_features["gt_path"] = mask_path
            clean_features["patient_id"] = patient_id

            all_features.append(clean_features)
            print(f"[{idx}] Extracted {len(clean_features)} features for patient {patient_id}")

        except Exception as e:
            print(f"[{idx}] Error processing {img_path}: {e}")

    # Saving the Dataframe
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"\nRadiomic features saved to: {output_csv} with shape {features_df.shape}")

    return features_df
import os
import json
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

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
        input_csv (str, optional): Path to CSV with columns 'exam_path', 'gt_path', 'glioma', 'ID'.
        output_csv (str, optional): Path to save extracted features CSV.
        bin_width (int): Bin width for intensity discretization.
        normalize (bool): Whether to normalize images before extraction.
        min_roi_size (int): Minimum ROI size.
        min_roi_dim (int): Minimum ROI dimension (2D or 3D).
        
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

    for idx, row in df.iterrows():
        img_path = row["exam_path"]
        mask_path = row["gt_path"]
        glioma_target = row["glioma"]
        patient_id = row["ID"]

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"[{idx}] Skipping missing file: {img_path} or {mask_path}")
            continue

        try:
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

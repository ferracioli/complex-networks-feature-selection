import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

def extract_radiomic_features(
    input_csv=None,
    output_csv=None,
    bin_width=40,
    normalize=False,
    min_roi_size=100,
    min_roi_dim=2
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

    # --- Default paths ---
    if input_csv is None:
        input_csv = "outputs/brats_africa/brats_africa_paths.csv"
    if output_csv is None:
        output_csv = "outputs/brats_africa/radiomic_features_brats_africa.csv"

    # Load CSV
    df = pd.read_csv(input_csv)

    # --- Configure extractor ---
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

    print("✅ Enabled all PyRadiomics features and image types.")

    # --- Feature extraction ---
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

    # --- Save DataFrame ---
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"\n✅ Radiomic features saved to: {output_csv}")
    print(f"Extracted DataFrame shape: {features_df.shape}")

    return features_df


# if __name__ == "__main__":
    # Example 1: extract all features
    # extract_radiomic_features()

    # Example 2: only specific feature classes
    # extract_radiomic_features(feature_classes=["firstorder", "glcm"])

    # Example 3: only specific features
    # extract_radiomic_features(feature_names={
    #     "firstorder": ["Energy", "TotalEnergy"],
    #     "glcm": ["Contrast", "Idmn"]
    # })


# import pandas as pd
# import SimpleITK as sitk
# from radiomics import featureextractor
# import os
# import nibabel as nib

# def extract_radiomic_features():
#     # Input CSV
#     input_csv = "outputs/brats_africa/brats_africa_paths.csv"
#     output_csv = "outputs/brats_africa/radiomic_features_brats_africa.csv"
#     df = pd.read_csv(input_csv)

#     extractor = featureextractor.RadiomicsFeatureExtractor()
#     extractor.settings['binWidth'] = 40
#     extractor.settings['normalizeScale'] = 1
#     extractor.settings['minimumROIDimensions'] = 2
#     extractor.settings['minimumROISize'] = 100 ## ERA 10
#     extractor.settings['enableMetadata'] = False
#     extractor.settings['normalize'] = False
#     extractor_settings = extractor.settings

#     # extractor = featureextractor.RadiomicsFeatureExtractor()
#     extractor.settings.update(extractor_settings)
#     extractor.disableAllFeatures()

#     extractor.enableFeaturesByName(
#         firstorder=['Energy', 'TotalEnergy'],
#         glcm=['JointAverage', 'Autocorrelation', 'Idmn'],
#         glrlm=['LowGrayLevelRunEmphasis', 'ShortRunLowGrayLevelEmphasis', 'HighGrayLevelRunEmphasis'],
#         gldm=['LowGrayLevelEmphasis', 'HighGrayLevelEmphasis', 'SmallDependenceEmphasis', 'SmallDependenceHighGrayLevelEmphasis', 'SmallDependenceLowGrayLevelEmphasis'],
#         ngtdm=['Contrast']
#     )

#     # Collect results
#     all_features = []

#     for idx, row in df.iterrows():
#         img_path = row["exam_path"]
#         mask_path = row["gt_path"]
#         glioma_target = row["glioma"]
#         patient_id = row["ID"]

#         if not os.path.exists(img_path) or not os.path.exists(mask_path):
#             print(f"Skipping missing file at index {idx}")
#             continue

#         try:

#             # Load image & mask
#             image = sitk.ReadImage(img_path)
#             mask = sitk.ReadImage(mask_path)

#             # Extract features
#             features = extractor.execute(image, mask)

#             # Keep only feature values (remove diagnostics)
#             clean_features = {k: v for k, v in features.items() if k.startswith("original")}
#             clean_features["glioma"] = glioma_target
#             clean_features["exam_path"] = img_path
#             clean_features["gt_path"] = mask_path
#             clean_features["patient_id"] = patient_id

#             all_features.append(clean_features)

#             print(f"Extracted features for {img_path}")

#         except Exception as e:
#             print(f"Error at {img_path}: {e}")

#     # Convert to DataFrame
#     features_df = pd.DataFrame(all_features)

#     # Save to CSV
#     features_df.to_csv(output_csv, index=False)
#     print(f"Features saved to {output_csv}")


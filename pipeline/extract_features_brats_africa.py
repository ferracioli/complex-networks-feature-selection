import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import os
import nibabel as nib

def main():
    # Input CSV
    input_csv = "brats_africa_paths.csv"
    output_csv = "radiomic_features_brats_africa.csv"
    df = pd.read_csv(input_csv)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['binWidth'] = 40
    extractor.settings['normalizeScale'] = 1
    extractor.settings['minimumROIDimensions'] = 2
    extractor.settings['minimumROISize'] = 100 ## ERA 10
    extractor.settings['enableMetadata'] = False
    extractor.settings['normalize'] = False
    extractor_settings = extractor.settings

    # extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings.update(extractor_settings)
    extractor.disableAllFeatures()

    extractor.enableFeaturesByName(
        firstorder=['Energy', 'TotalEnergy'],
        glcm=['JointAverage', 'Autocorrelation', 'Idmn'],
        glrlm=['LowGrayLevelRunEmphasis', 'ShortRunLowGrayLevelEmphasis', 'HighGrayLevelRunEmphasis'],
        gldm=['LowGrayLevelEmphasis', 'HighGrayLevelEmphasis', 'SmallDependenceEmphasis', 'SmallDependenceHighGrayLevelEmphasis', 'SmallDependenceLowGrayLevelEmphasis'],
        ngtdm=['Contrast']
    )

    # Collect results
    all_features = []

    for idx, row in df.iterrows():
        img_path = row["exam_path"]
        mask_path = row["gt_path"]
        glioma_target = row["glioma"]
        patient_id = row["ID"]

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Skipping missing file at index {idx}")
            continue

        try:

            # Load image & mask
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)

            # Extract features
            features = extractor.execute(image, mask)

            # Keep only feature values (remove diagnostics)
            clean_features = {k: v for k, v in features.items() if k.startswith("original")}
            clean_features["glioma"] = glioma_target
            clean_features["exam_path"] = img_path
            clean_features["gt_path"] = mask_path
            clean_features["patient_id"] = patient_id

            all_features.append(clean_features)

            print(f"Extracted features for {img_path}")

        except Exception as e:
            print(f"Error at {img_path}: {e}")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Save to CSV
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    main()

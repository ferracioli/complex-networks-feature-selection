import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import os
import nibabel as nib

def main():
    # Input CSV
    input_csv = "pipeline/upenn_paths.csv"
    df = pd.read_csv(input_csv)

    # Initialize PyRadiomics extractor (with default settings)
    # extractor = featureextractor.RadiomicsFeatureExtractor()

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
        mgmt_status = row["MGMT"]

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Skipping missing file at index {idx}")
            continue

        try:
            # This commented version does not work
            # nii_image = nib.load(img_path)
            # image_data = nii_image.get_fdata()

            # nii_mask = nib.load(mask_path)
            # mask_data = nii_mask.get_fdata()

            # self.affine = nii_image.affine

            # self.voxel_sizes = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
            # self.voxel_sizes = nii_image.header.get_zooms()

            # Load image & mask
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)

            # Extract features
            features = extractor.execute(image, mask)

            # Keep only feature values (remove diagnostics)
            clean_features = {k: v for k, v in features.items() if k.startswith("original")}
            clean_features["MGMT"] = mgmt_status
            clean_features["exam_path"] = img_path
            clean_features["gt_path"] = mask_path

            all_features.append(clean_features)

            print(f"Extracted features for {img_path}")

        except Exception as e:
            print(f"Error at {img_path}: {e}")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Save to CSV
    output_csv = "radiomic_features.csv"
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    main()

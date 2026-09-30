# from pipeline.extract_features import generate_exam_dataframe, extract_radiomic_features
from pipeline.model_evaluation import model_benchmarking

def main():

    # This function is responsible by triggering the benchmark for every dataset
    # References for the datasets:
    # BraTS Africa: https://www.cancerimagingarchive.net/collection/brats-africa/
    # NSCLC: https://openradiomics.org/brats-2020/
    # Radiomics LGG: https://www.kaggle.com/datasets/knamdar/radiomics-for-lgg-dataset

    # NOTE: BraTS Africa does not have radiomic features available, you must use the comment section
    # To extract radiomic features from its dataset

    extract_features = False
    datasets = ["radiomics_lgg", "four_class_ncsls", "brats_africa"]

    for dataset in datasets:

        # NOTE: Pyradiomics requires installing visual studio C++ before using it
        # if extract_features and dataset == "brats_africa":
        #     # 1) generate dataframe
        #     print("Generating the list of available images")
        #     generate_exam_dataframe(dataset=dataset)

        #     # 2) extract_features
        #     print("Extracting radiomic features")
        #     extract_radiomic_features(dataset=dataset)

        # 3) run model_evaluation comparing with the complex network selection
        print("Benchmarking models")
        model_benchmarking(dataset=dataset)

if __name__ == "__main__":
    main()
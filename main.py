# from pipeline.extract_features import generate_exam_dataframe, extract_radiomic_features
from pipeline.model_evaluation import model_benchmarking

def main():

    extract_features = False
    datasets = ["radiomics_lgg", "nsclc", "four_class_nsclc", "brats_africa"]

    for dataset in datasets:

        # Only BraTS Africa requires feature extraction
        # If you are not extracting features with pyradiomics this can be commented
        # Pyradiomics requires installing visual studio C++ before using it
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
import os
import pandas as pd
import json

# Loading the config json
with open('input/config.json', 'r') as file:
    config = json.load(file)

# Remotion_list is a variable used to continue the mapping if it gets interrupted
def generate_exam_dataframe(max_items=None, remotion_list=[], dataset="brats_africa"):
    # D:\data\BraTS-Africa\51_OtherNeoplasms\BraTS-SSA-00018-000
    # glioma_path = r'D:\data\BraTS-Africa\95_Glioma'
    # others_path = r'D:\data\BraTS-Africa\51_OtherNeoplasms'
    # clinical_csv = r'D:\data\BraTS-Africa\demography.csv'
    # output_csv = "outputs/brats_africa/brats_africa_paths.csv"
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


# df = generate_exam_dataframe()  # por exemplo, limitar a 100 exames
# generate_exam_dataframe()


# import os
# import pandas as pd

# def generate_exam_dataframe(max_items=None, remotion_list=[], dataset="brats_africa"):
#     # Caminhos
#     # D:\data\BraTS-Africa\51_OtherNeoplasms\BraTS-SSA-00018-000
#     glioma_path = r'D:\data\BraTS-Africa\95_Glioma'
#     others_path = r'D:\data\BraTS-Africa\51_OtherNeoplasms'
#     clinical_csv = r'D:\data\BraTS-Africa\demography.csv'
#     output_csv = "outputs/brats_africa/brats_africa_paths.csv"

#     # Carregar CSV clínico
#     clinical_df = pd.read_csv(clinical_csv, sep=",")
#     print(clinical_df.columns)
#     # Ajuste: dependendo da coluna, troque "Subject_ID" pelo nome certo no CSV
#     if "ID" in clinical_df.columns:
#         id_col = "ID"
#     else:
#         id_col = clinical_df.columns[0]  # pega a primeira se não soubermos

#     exam_list = []

#     # Glioma images
#     if os.path.exists(glioma_path):
#         exam_folders = os.listdir(glioma_path)

#         for exam in exam_folders:
#             if max_items and len(exam_list) >= max_items:
#                 break

#             if exam in remotion_list:
#                 print(f"{exam} já foi verificado, ignorando...")
#                 continue

#             exam_file = os.path.join(glioma_path, exam, f"{exam}-t2f.nii.gz")
#             if not os.path.isfile(exam_file):
#                 continue

#             # Manual segmentation
#             segm_file = os.path.join(glioma_path, exam, f"{exam}-seg.nii.gz")

#             gt_file = segm_file

#             exam_list.append({
#                 "exam_id": exam,
#                 "exam_path": exam_file,
#                 "gt_path": gt_file
#             })

#     # Non Glioma images
#     if os.path.exists(others_path):
#         exam_folders = os.listdir(others_path)
#         print(exam_folders)

#         for exam in exam_folders:
#             if max_items and len(exam_list) >= max_items:
#                 break

#             if exam in remotion_list:
#                 print(f"{exam} já foi verificado, ignorando...")
#                 continue

#             exam_file = os.path.join(others_path, exam, f"{exam}-t2f.nii.gz")
#             if not os.path.isfile(exam_file):
#                 continue

#             # Manual segmentation
#             segm_file = os.path.join(others_path, exam, f"{exam}-seg.nii.gz")

#             gt_file = segm_file

#             exam_list.append({
#                 "exam_id": exam,
#                 "exam_path": exam_file,
#                 "gt_path": gt_file
#             })

#     # Converte lista para DataFrame
#     exams_df = pd.DataFrame(exam_list)

#     # Faz o merge com info clínica
#     merged_df = clinical_df.merge(exams_df, left_on=id_col, right_on="exam_id", how="inner")

#     # Salva CSV final
#     merged_df.to_csv(output_csv, index=False)

#     print(f"Arquivo salvo em: {output_csv}")
#     print(merged_df.shape)
#     return merged_df


# # df = generate_exam_dataframe()  # por exemplo, limitar a 100 exames
# # generate_exam_dataframe()

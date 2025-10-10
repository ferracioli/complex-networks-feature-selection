# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# def generate_exam_list(max_items, remotion_list = []):

#     # Caminho para o diretório 'data' no disco D:
#     base_path = r'D:\data\PKG - UPENN-GBM-NIfTI\UPENN-GBM\NIfTI-files'
#     # o csv está em D:\data\PKG - UPENN-GBM-NIfTI\UPENN-GBM\UPENN-GBM_clinical_info_v2.1.csv
#     # (o separador é a vírgula)
#     # fazer um join do ID desse csv com o basename dos arquivos, e adicionar as colunas
#     # exam_path e gt_path
#     # As imagens seguem esse template de filename UPENN-GBM-00001_11_automated_approx_segm.nii.gz
#     # O csv final deve ser salvo localmente com o nome de upenn_paths.csv
#     exam_list = []

#     exam_path = base_path + '\images_structural'

#     # Verifica se o diretório existe
#     if os.path.exists(exam_path):
#         # Lista os arquivos e pastas no diretório
#         exam_folders = os.listdir(exam_path)
        
#         # For each exam, searches for the segmentation
#         for exam in exam_folders:

#             if len(exam_list) == max_items:
#                 break

#             # Pula exemplos já processados
#             if exam in remotion_list:
#                 print(f"{exam} já foi verificado, ignorando...")
#                 continue

#             exam_file = f"{exam_path}\{exam}\{exam}_FLAIR.nii.gz"
#             if not os.path.isfile(exam_file):
#                 continue

#             # Checking the automatic segmentation
#             automatic_folder = "automated_segm"
#             automatic_segm_path = f"{base_path}\{automatic_folder}"
#             automatic_segm_file = f"{automatic_segm_path}\{exam}_automated_approx_segm.nii.gz"
#             if os.path.isfile(automatic_segm_file):
#                 # Append the exam and continues the search
#                 new_exam = {
#                     "exam_id": exam,
#                     "exam_path": exam_file,
#                     "exam_ground_truth": automatic_segm_file
#                 }
#                 exam_list.append(new_exam)
#                 continue

#             # Checking the manual segmentation
#             segm_path = base_path + '\images_segm'
#             segm_file = f"{segm_path}\{exam}_segm.nii.gz"
#             if os.path.isfile(segm_file):
#                 # Append the exam and continues the search
#                 new_exam = {
#                     "exam_id": exam,
#                     "exam_path": exam_file,
#                     "exam_ground_truth": segm_file
#                 }
#                 exam_list.append(new_exam)
#                 continue

#     return exam_list

import os
import pandas as pd

def generate_exam_dataframe(max_items=None, remotion_list=[]):
    # Caminhos
    base_path = r'D:\data\PKG - UPENN-GBM-NIfTI\UPENN-GBM\NIfTI-files'
    clinical_csv = r'D:\data\PKG - UPENN-GBM-NIfTI\UPENN-GBM_clinical_info_v2.1.csv'
    output_csv = "upenn_paths.csv"

    # Carregar CSV clínico
    clinical_df = pd.read_csv(clinical_csv, sep=",")
    # Ajuste: dependendo da coluna, troque "Subject_ID" pelo nome certo no CSV
    if "ID" in clinical_df.columns:
        id_col = "ID"
    else:
        id_col = clinical_df.columns[0]  # pega a primeira se não soubermos

    exam_path = os.path.join(base_path, "images_structural")
    exam_list = []

    if os.path.exists(exam_path):
        exam_folders = os.listdir(exam_path)

        for exam in exam_folders:
            if max_items and len(exam_list) >= max_items:
                break

            if exam in remotion_list:
                print(f"{exam} já foi verificado, ignorando...")
                continue

            exam_file = os.path.join(exam_path, exam, f"{exam}_FLAIR.nii.gz")
            if not os.path.isfile(exam_file):
                continue

            # Automatic segmentation
            automatic_folder = "automated_segm"
            automatic_segm_file = os.path.join(base_path, automatic_folder, f"{exam}_automated_approx_segm.nii.gz")

            # Manual segmentation
            segm_path = os.path.join(base_path, "images_segm")
            segm_file = os.path.join(segm_path, f"{exam}_segm.nii.gz")

            gt_file = None
            if os.path.isfile(automatic_segm_file):
                gt_file = automatic_segm_file
            elif os.path.isfile(segm_file):
                gt_file = segm_file
            else:
                continue  # pula se não houver GT

            exam_list.append({
                "exam_id": exam,
                "exam_path": exam_file,
                "gt_path": gt_file
            })

    # Converte lista para DataFrame
    exams_df = pd.DataFrame(exam_list)

    # Faz o merge com info clínica
    merged_df = clinical_df.merge(exams_df, left_on=id_col, right_on="exam_id", how="inner")

    # Salva CSV final
    merged_df.to_csv(output_csv, index=False)

    print(f"Arquivo salvo em: {output_csv}")
    print(merged_df.shape)
    return merged_df


df = generate_exam_dataframe()  # por exemplo, limitar a 100 exames
# IMPORTANT
# this is just a template from my previous project
from pipeline.select_supervoxels import SupervoxelSelection
from pipeline.segment_region import SegmentationGenerator
from pipeline.evaluate_segmentation import EvaluateSegmentation
from pipeline.extract_features_supervoxel import SupervoxelFeatureExtractor
from pipeline.utils import generate_exam_list, calculate_metrics
import json
import time
import os

def main():

    # Carrega o arquivo inicial de configurações base
    with open('pipeline/config.json', 'r') as file:
        config = json.load(file)

    n_segments = 256
    compactness = 0.01
    sigma = 1
    nb_exams = 10
    check_nodes = False

    # Se já existe um json de resultados, continua a partir dele
    if os.path.exists(f"{config['data']['paths']['output_dir']}results.json"):
        with open(f"{config['data']['paths']['output_dir']}results.json", 'r') as file:
            results = json.load(file)

        # Desconsiderar exames que já foram segmentados
        to_remove = [key for key in results.keys() if key != "params"]
        config["data"]["exam_list"] = generate_exam_list(nb_exams, remotion_list=to_remove)

    # Cria uma chave para metadados do experimento
    else:
        results = {}
        results["params"] = {}
        results["params"]["n_segments"] = n_segments
        results["params"]["compactness"] = compactness
        results["params"]["sigma"] = sigma
        config["data"]["exam_list"] = generate_exam_list(nb_exams)

    param_str = f"{str(compactness)}-{str(sigma)}"

    for exam in config['data']['exam_list']:

        # Adiciona o exame atual na listagem de resultados
        results[exam["exam_id"]] = {}
        start_time = time.time()

        # Passo 1: extrair os supervoxels do exame
        supervoxels = SupervoxelFeatureExtractor(
            n_segments = n_segments, 
            compactness = compactness, 
            sigma = sigma,
            params = config["params"],
            exam = exam,
            exam_paths = config["data"]["paths"]
        )
        tot_clusters, connections, image_shape, lista_supervoxels, affine = supervoxels.run()

        # Passo 2: selecionar os supervoxels mais prováveis
        clustering = SupervoxelSelection(
            exam = exam,
            exam_paths = config["data"]["paths"],
            params = config["params"],
            n_segments = n_segments,
            connections = connections,
            param_str = param_str
        )
        nb_features, selected_clusters, used_features = clustering.run()
        print(selected_clusters)

        results[exam["exam_id"]]['found clusters'] = str(tot_clusters)
        results[exam["exam_id"]]['nb features'] = nb_features

        # Passo 3: converter os supervoxels em uma imagem 3D
        segmenter = SegmentationGenerator(
            exam = exam,
            exam_paths = config["data"]["paths"],
            params = config["params"],
            selected_clusters = selected_clusters,
            image_shape = image_shape,
            affine = affine
        )
        segmenter.run()

        # Passo 4: comparar com o tumor original
        # evaluation = EvaluateSegmentation(
        #     exam = exam,
        #     exam_paths = config["data"]["paths"],
        #     selected_clusters = selected_clusters
        # )
        evaluation = EvaluateSegmentation(
            exam = exam,
            exam_paths = config["data"]["paths"]["output_dir_exams"],
            selected_clusters = selected_clusters
        )
        iou, precision, volume = evaluation.run()
        end_time = time.time()
        execution_time = end_time - start_time

        # Armazena o desempenho
        results[exam["exam_id"]]["Selected supervoxels"] = str(selected_clusters)
        results[exam["exam_id"]]["IoU"] = f"{iou:.4f}"
        results[exam["exam_id"]]["Precision"] = f"{precision:.4f}"
        results[exam["exam_id"]]["Volume"] = f"{volume:.4f}"
        results[exam["exam_id"]]["Total time(sec)"] = f"{execution_time:.4f}"

        # Seção de código que verifica quais são os supervoxels que permitem o melhor resultado
        if check_nodes:
            valid_nodes = []
            # Realiza a segmentação para cada supervoxel
            for c in lista_supervoxels:

                selected_clusters = [ c ]

                # Parte diretamente para a segmentação
                segmenter = SegmentationGenerator(
                    exam = exam,
                    exam_paths = config["data"]["paths"],
                    params = config["params"],
                    selected_clusters = selected_clusters,
                    image_shape = image_shape,
                    affine = affine
                )
                segmenter.run()

                # Avalia o supervoxel individualmente
                evaluation = EvaluateSegmentation(
                    exam = exam,
                    exam_paths = config["data"]["paths"],
                    selected_clusters = selected_clusters
                )
                # Se possui um IoU relevante, armazena como metadado adicional
                iou, precision, volume = evaluation.run()
                if float(iou) >= 0.0400 and float(precision) >= 0.5:
                    valid_nodes.append(c)
                    results[exam["exam_id"]][str(c)] = f"iou {iou:.4f}, precision {precision:.4f}"
            results[exam["exam_id"]]["valid_nodes"] = str(valid_nodes)

        # Atualiza o arquivo de resultados a cada iteração
        with open(f"{config['data']['paths']['output_dir']}results.json", 'w') as fp:
            json.dump(results, fp, indent=4)

    calculate_metrics(config['data']['paths']['output_dir'], source="")

if __name__ == "__main__":
    main()
## Complex Networks Feature Selection

### Setup
This repository was implemented with the following instalation steps:
-1: Install python(used version: 3.9.13)

-2: Build a virtual environment inside your directory:
Only run the command during your first run, according to the python command
`python -m venv venv`
`py -3.9 -m venv venv`

-3: Activate your virtual environment:
`./venv/Scripts/activate`

-4: During the first run, install the requirements:
`pip install -r requirements.txt`

Or, if you want to force the reinstalation ignoring cache:
`pip install --no-cache-dir -r requirements.txt`

-5: In case you are including GFSIR to your benchmark, please insert the source code
at /pipeline folder. The repository can be found at https://github.com/hmMed22/GFSIR/tree/main
In case of not using GFSIR, please comment its results.append(run_eval(model_data)) section.

-6: Run the main file
`python .\main.py`

### Supplementary information
During the setup for the experiment, an external drive was used to store all the datasets used during the project. This may not impact the running of the application, and you can store the dataset at the same disk as the source code, as long as you have enough storage and map the root folder.

The first dataset used was the BraTS Africa, and the URL for loading the dataset can be found at: https://www.cancerimagingarchive.net/collection/BraTS-Africa/

It is important to notice that the class dataframe at this dataset was manually mapped, and the mapped file will be available at the repository[TODO]. Before starting the code execution, you must paste the initial mapping file to your dataset.



Interesting repos for datasets:
https://github.com/pwoznicki/RadiomicsHub/tree/master/radhub
https://www.cancerimagingarchive.net/nih-controlled-data-access-policy/


#### Datasets

1) Brats Aftica
https://pubs.rsna.org/doi/10.1148/ryai.240528
https://www.cancerimagingarchive.net/collection/BraTS-Africa/
– Tipo de imagem: MRI multiparamétrico de cérebro (T1, T1Gd, T2, T2-FLAIR)
– Tarefa original: segmentação de gliomas em sub-regiões (Enhancing tumor, Non-enhancing core, etc.).
– Micro-ambiente dos dados: dados clínicos heterogêneos, scanners diferentes (baixa resolução típica de contextos com infraestrutura limitada).
Brats Africa:
Número base de features radiômicas: 1409
104 treino / 26 teste
Runtime:
Redes complexas tiveram as melhores performances(bal_acc em torno de 0.8), thresh entre 0 e 0.15
algoritmos classicos tiveram bal_acc entre 0.63 e 0.66 com um tempo MUITO menor
piores casos de redes complexas tiveram desempenho de 0.44, sem um padrão claro de por que a pontuação foi baixa(não consegui definir qual das configs causou perda de desempenho, mas provavelmente escolheu features aleatórias)
rede complexas tiveram um tempo máximo de 280 segundos(thresh 0). O tempo mais rápido das redes foi de 16s para thresh 0.7
Target glioma(binário) com essa distribuição:
false 43
true 87
após rodar com degree rank e dropando colunas estáveis:
sem tanta mudança na performance
tempo de redes complexas foi de 15 a 244s(mudança breve no tempo máximo)
> no grafico de acuracia balanceada por threshold, parece ser tipo um u o formato do gráfico(threshs baixos e altissimos) tiveram a melhor performance


2) radiomics LGG
https://www.kaggle.com/datasets/knamdar/radiomics-for-lgg-dataset?select=train.csv
https://link.springer.com/article/10.1186/s12880-025-01855-2
Esse dataset é uma curadoria tabular de features radiômicas já extraídas, publicada no Kaggle.
No entanto, não existe um artigo científico estritamente “original” dedicado a essa versão específica publicada no Kaggle
– Imagens de cabeça (MRI) de gliomas; foco em LGG vs. HGG (no artigo original BraTS 2020).
– Features radiômicas heterogêneas foram extraídas via PyRadiomics no estudo Open-Radiomics (combinações de sequências, vários parâmetros de extração).
– Classificação binária de pacientes com glioma de baixo grau (LGG) vs. outros grupos (no contexto Open-Radiomics, HGG vs. LGG)
105 pacientes com o target já binarizado
radiomics_lgg(classificação binária):
Número inicial de features: 640
105 treino / 45 teste - runtime 14min27s para todas as combinações
Algoritmos classicos ficaram com a melhor performance(bal_acc entre 0.65 e 0.66) com um tempo bem menor
Redes complexas tiveram um tempo máximo de 62s(thresh 0), casos mais rápidos duraram 3s(thresh 0.7)
Melhores modelos de redes complexas tiveram bal_acc entre 0.63 e 0.61, tendendo a um threshold mais alto e ao label propagation
Piores casos de redes complexas ficaram com bal_acc entre 0.35 e 0.38
Target Mutacion(binário). Teve a seguinte distribuição:
0 38
1 67
Ficou assim depois de dropar features muito estáveis:
Initial number of features: 545
[depois de rodar de novo adicionando degree rank]: thresholds altos tiveram os melhores resultados nas redes, de modo geral os modelos clássicos ganharam. Tempo radiomico foi de 3 a 29s(melhora no tempo máximo)
> esse foi o único em que o threshold alto foi mais correlacionado com a acurácia

website Open radiomics
https://openradiomics.org/
BraTS 2020-OpenRadiomics
T1 contrast-enhanced sequence and the union of Necrotic and the non-enhancing tumor core subregions resulted in the highest AUROCs
dataset binário(group label com LGG-76 no 0 e 293 no HGG-1)
features já extraídas, várias categorias de width

3) TCIA NSCLC-OpenRadiomics
https://arxiv.org/abs/2207.14776
esse é de pulmão
esse aqui possivelmente é o melhor, histology tem varias classes:
histology: 152 squamous, 51 adenor..., 114 large cell, 63 nos, 42 null
tbm tem colunas de stage
NSCLC:
– Imagens de tórax (CT) de pacientes com NSCLC (non-small cell lung cancer)
– Características radiômicas extraídas para aplicações de classificação e sobrevivência
1688 features no início
336 train / 85 test - runtime 1h02m23s para todas as combinações
depois de adicionar novo método foi para 1h44m36s
redes complexas tiveram melhor performance (bal_acc entre 0.47 e 0.48)
algoritmos clássicos tiveram performance de bal_acc entre 0.43 e 0.44, e o lasso ficou em 0.38
thresh 0.6 e label propagation foi sempre o melhor
piores casos de redes complexas tiveram bal_acc de 0.21(praticamente uma escolha aleatória)
tempo máximo de 370s (thresh 0), e 8s para thresh 0.7. Nesse caso em específico, o lasso durou 38s, o que é surpreendente pois as redes sempre eram muito mais lentas
Target clinical_T_stage(multiclasse). Os valores são:
5.0 2
1.0 93
2.0 156
3.0 53
4.0 117
Após adicionar o degree_rank, não teve muita mudança nas scores
todos os modelos demoraram pra rodar
tempo de redes foi de 16 a 342s(melhora muito baixa)
> pelo gráfico, parece que a performance é até que relacionada entre threshold e acurácia, com um pico entre 0.3 e 0.4

[NÃO USADOS]
BraTS 2023-OpenRadiomics
esse aqui parece ser sobre o MGMT
tem 276 com mgmt 0 e 301 com mgmt 1

Opção 2 ReMIND(não usado ainda): https://www.cancerimagingarchive.net/collection/remind/
Menos desbalanceado entre classes, mas só tem 114 pacientes

Opção 3: Pet radiomics(não usado ainda): https://www.kaggle.com/competitions/pet-radiomics-challenges/data?select=Training.zip
mais de 400 pacientes, mas os exames são em dcm
The ultimate goal will be the development of an algorithm that yields the probability of local tumor control in oropharynx cancer patients who received definitive radiation treatment

### Sample dataset
Patients with original_shape_MajorAxisLength > 55 and original_shape_MeshVolume > 65000 -> target = 1
Otherwise -> target = 0


Parâmetros usados:
"thresholds": [0.0, 0.15, 0.3, 0.45, 0.6, 0.7],
"link_methods": ["cosine", "spearman", "pearson", "rho_distance"],
"cn_selectors": ["lp", "pr"],
"eigen_options": [false],
"classical_selectors": ["lasso", "information_gain", "gini"]





SELECT cast(d->>'clinical.T.Stage' as varchar), count(cast(d->>'clinical.T.Stage' as varchar))
FROM data d
group by cast(d->>'clinical.T.Stage' as varchar);
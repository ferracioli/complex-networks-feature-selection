## Complex Networks Feature Selection

### Setup
This repository was implemented with the following instalation steps:
-1: Install python(used version: 3.9.13)

-2: Build a virtual environment inside your directory:
Only run this command during your first run, according to the python command
`python -m venv venv`
`py -3.9 -m venv venv`

-3: Activate your virtual environment(might have slight differences if used in Windows or Linux):
`./venv/Scripts/activate`

-4: During the first run, install the requirements. To avoid circular dependencies, please install numpy and pyradiomics first and then the required libraries. Pyradiomics is only required if you are also extracting features from 3D images:
`pip install numpy<2 wheel setuptools`
`pip install pyradiomics`
`pip install -r requirements.txt`

Or, if you want to force the reinstalation ignoring cache:
`pip install --no-cache-dir -r requirements.txt`

-5: In case you are including GFSIR to your benchmark, please insert the source code
at /pipeline folder. The repository can be found at https://github.com/hmMed22/GFSIR/tree/main
In case of not using GFSIR, please comment its results.append(run_eval(model_data)) section.

-6: Update the main file with the dataset you want to run(assuming you have its folder also configured). Then run the model evaluation:
`python .\main.py`

### Supplementary information
During the setup for the experiment, an external drive was used to store all the datasets used during the project. This may not impact the running of the application, and you can store the dataset at the same disk as the source code, as long as you have enough storage and map the root folder.

This decision was done for the BraTS Africa dataset, since it holds a huge amount of data. This can be skipped if you do not intend to extract radiomic features.The  URL for loading the dataset can be found at: https://www.cancerimagingarchive.net/collection/BraTS-Africa/

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

-5: Run the main file
`python .\main.py`

### Supplementary information
During the setup for the experiment, an external disk was used to store all the datasets used during the project. This may not impact the running of the application, and you can store the dataset at the same disk as the source code, as long as you have enough storage and map the root folder.

The first dataset used was the BraTS Africa, and the URL for loading the dataset can be found at: https://www.cancerimagingarchive.net/collection/BraTS-Africa/

It is important to notice that the class dataframe at this dataset was manually mapped, and the mapped file will be available at the repository[TODO]. Before starting the code execution, you must paste the initial mapping file to your dataset.
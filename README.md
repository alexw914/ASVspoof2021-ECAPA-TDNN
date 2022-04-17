# ASVspoof 2021 LA Spoof Speech Detection System 

A voice spoofing detection system, based on ECAPA-TDNN. This project mainly follow these projects. [Project1](https://github.com/pabdzadeh/voice-spoof-detection-system)

## Installation

First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ pip install -r requirements.txt
```

### Dataset
Our experiments are done in the logical access (LA) partition of the ASVspoof 2021 dataset, which can can be downloaded from [here](https://www.asvspoof.org/https://datashare.is.ed.ac.uk/handle/10283/3336). Training data is ASVspoof 2019 [here](https://datashare.is.ed.ac.uk/handle/10283/3336). Put the LA folder in data

### Preprocess
To get the feature data, you can follow the preprocess file, or you can use other tools such as torchaudio to get the LFCC feature. This method refers to ASVspoof 2019 baseline. After preprocess， put the feature data folder in this project. (After preprocess,you can find it in LA folder in dataset folder(LA), if you don't want make preporcess before training, you can change dataset_loader.py in tools folder. You can compute LFCC feature while training)

### Training
To train the model run:
```
python main.py
```

### Test
All the scorefile will be store in scores folder. This algorithm will test EER and min-tDCF of progres set in every epoch.

You should change the folder in find_best.py to find the best checkpoints in progress phase And you can test it in eval phase.

To test the model run:
```
python test.py --model_index $(best epoch)
```


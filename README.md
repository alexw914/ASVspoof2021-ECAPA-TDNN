# ASVspoof 2021 LA Spoof Speech Detection System 

A voice spoofing detection system, based on ECAPA-TDNN. This project mainly follow these [project](https://github.com/pabdzadeh/voice-spoof-detection-system). If you want do some research or improve the architecture, you can refer this [repo](https://github.com/yzyouzhang/ASVspoof2021_AIR). It provides the way of augment and some tips to train a more robust network.

## Installation

First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ pip install -r requirements.txt
```

### Dataset
This project is for (LA) partition of the ASVspoof 2021 challenge, which can can be downloaded from [here](https://www.asvspoof.org/https://datashare.is.ed.ac.uk/handle/10283/3336). Training data is ASVspoof 2019 [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Preprocess
To get the feature data, you can follow the preprocess.py file, or you can use other tools such as torchaudio to get the LFCC feature. This method refers to ASVspoof 2019 baseline. After preprocess， put the feature_data folder in this project. (After preprocess, you can find it in LA folder in dataset folder(LA), if you don't want make preporcess before training, you can change dataset_loader.py in tools folder. You can compute LFCC feature while training)

### Training
To train the model run:
```
python main.py
```

### Test
All the scorefiles will be store in scores folder. This algorithm will test EER and min-tDCF of progres set in every epoch.

You should change the score_file_root in find_best.py to find the best checkpoints in progress phase. And you can test it in eval phase.

To test the model run:
```
python test.py --model_index $(best epoch)
```
### Warning
The result of this project may not very well because in ASVspoof 2021 challenge， the problem of codec transmisson and compression is introduced. There are many ways you can solve it. This [repo](https://github.com/alexw914/ASVspoof_CM) and can help you.

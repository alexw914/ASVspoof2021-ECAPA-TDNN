from random import randint, random
from time import time
import torch
import collections
import os, warnings, librosa, torchaudio
import torch.nn.functional as F
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

##########################################################################
####### Put the raw waveform in data folder. The feature data folder is LFCC  
####### feature data, you can get LFCC feature by preprocess.py.
####### Except flac folder, the other data is augmented by FFMPEG and SoX.
##########################################################################

LOGICAL_DATA_ROOT = './data/'
LOGICAL_FEAT_ROOT = "./feature_data/"

ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'file_name','key'])


warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

# For LA augmentation
transmission_aug = ["flac"
            # ,  
            # 'amr_nb-br12.20k','amr_nb-br4.75k','amr_nb-br7.4k',
            # 'amr_wb-br15.85k','amr_wb-br23.85k','amr_wb-br6.6k',
            # 'g711-alaw','g711-mulaw',
            # 'g722-br128k','g722-br48k','g722-br58k','g722-br64k',
            # 'gsm-8k',
            # 'opus-br160k','opus-br80k'
            ]

# For DF augmentation
compression_aug = ['flac'
            # ,'m4a-br142k','m4a-br20k','m4a-br96k',
            # 'mp2-sr16k','mp2-sr24k','mp2-sr48k',
            # 'mp3-br64k','mp3-sr16k','mp3-sr8k',
            # 'ogg-br48k','ogg-br80k',
            ]

# device_aug = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
#              'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
#              'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000'
#              ]

status_type = ["train", "dev", "eval", "eval2021", "df2021"]



class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    """
    set: train, eval2021-progress, eval2021-eval
    """

    def __init__(self, type = "train", feature_type='lfcc'):
        super(ASVDataset, self).__init__()

        feat_root = LOGICAL_FEAT_ROOT
        audio_root = LOGICAL_DATA_ROOT 
        self.type    = type
        self.feature_type = feature_type
        self.audio_root = audio_root
        self.feat_root  = feat_root

        self.protocols_fname = os.path.join(self.audio_root, self.type + '.protocol.txt')

        self.audio_dir = os.path.join(self.audio_root, self.type)
        lines = open(self.protocols_fname).readlines()
        self.files_meta = list(map(self._parse_line, lines))


    def __len__(self):
        return len(self.files_meta)

    def __getitem__(self, idx):
        idx = idx % len(self.files_meta)
        meta = self.files_meta[idx]

        if self.type == "train":

            aug_prob = random()
            #########################################
            ##### OPTION if you only need transmision
            ##### or compression aug_type set aug_prob
            #########################################
            aug_prob = 0.4
            if aug_prob <= 0.5:
                transmission_aug_type = randint(0,len(transmission_aug)-1)
                feat_path = os.path.join(self.feat_root, self.type, transmission_aug[transmission_aug_type], self.feature_type, meta.file_name+".pt")
                data_x = torch.load(feat_path)
            else:
                compression_aug_type = randint(0,len(compression_aug)-1)
                feat_path = os.path.join(self.feat_root, self.type, compression_aug[compression_aug_type], self.feature_type, meta.file_name+".pt")
                data_x = torch.load(feat_path)

            if data_x.shape[1] > 400:    
                start_frame = np.int64(random() * (data_x.shape[1]-400))
                data_x = data_x[:, start_frame: start_frame+400]
            if data_x.shape[1] < 400:
                data_x = repeat_padding_Tensor(data_x, 400)
       
        else:
            if self.type == "eval2021-eval" or self.type == "eval2021-progress":
                feat_path = os.path.join(self.feat_root, "eval2021", transmission_aug[0], self.feature_type, meta.file_name+".pt")
            else:
                feat_path = os.path.join(self.feat_root, "df2021", transmission_aug[0], self.feature_type, meta.file_name+".pt")
            data_x = torch.load(feat_path)

            if data_x.shape[1] < 750:
                data_x = repeat_padding_Tensor(data_x, 750)
            else:
                data_x = data_x[:,:750]    
        
        # data_x = data_x.unsqueeze(0)    
        data_y = float(meta.key)

        if self.type == "train":
            return data_x, data_y
        else:
            return data_x, data_y, self.files_meta[idx]

    def _parse_line(self, line):
        
        tokens = line.strip().split(' ')

        if self.type == "train":
            return ASVFile(speaker_id=tokens[0],
                        file_name=tokens[1],
                        key=int(tokens[4] == 'bonafide'))

        else:
            return ASVFile(speaker_id='',
                           file_name=tokens[1],
                           key=int(tokens[5] == 'bonafide')
                           )


def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

if __name__ == "__main__":
    train_set = ASVDataset("eval2021-eval")
    train_loader = DataLoader(train_set, 
                            batch_size=16, 
                            num_workers=4, 
                            shuffle=True, 
                            pin_memory=True,
                            )
    epoch = 0
    print(len(train_set))



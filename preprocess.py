import soundfile as sf
import pandas as pd
import numpy as np
import torch, torchaudio
import librosa
from scipy import signal
import os
from tqdm import tqdm
from random import random
import warnings
import torch.nn.functional as F
from rich import console
from rich.console import Console
from tools.feature_layers import LFCC

console = Console()

"""
Data processing for lfcc feature extraction
"""


warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')
### augtype and file_type list,
aug_type = ["flac"
            # ,  
            # 'amr_nb-br12.20k','amr_nb-br4.75k','amr_nb-br7.4k',
            # 'amr_wb-br15.85k','amr_wb-br23.85k','amr_wb-br6.6k',
            # 'g711-alaw','g711-mulaw',
            # 'g722-br128k','g722-br48k','g722-br58k','g722-br64k',
            # 'gsm-8k',
            # 'm4a-br142k','m4a-br20k','m4a-br96k',
            # 'mp2-sr16k','mp2-sr24k','mp2-sr48k',
            # 'mp3-br64k','mp3-sr16k','mp3-sr8k',
            # 'ogg-br48k','ogg-br80k',
            # 'opus-br160k','opus-br80k'
            ]
file_type =["flac"
            # ,
            # "wav", "wav", "wav",
            # "wav", "wav", "wav",
            # "wav", "wav",
            # "wav", "wav", "wav","wav",
            # "wav", 
            # "m4a", "m4a", "m4a",
            # "mp2", "mp2", "mp2",
            # "mp3", "mp3", "wav", 
            # "ogg", "ogg",
            # "opus","opus"
        ]
status_type = ["train", "dev", "eval", "eval2021", "df2021"]


def delta(x):
    """ By default
    input
    -----
    x (batch, Length, dim)

    output
    ------
    output (batch, Length, dim)

    Delta is calculated along Length
    """
    length = x.shape[1]
    output = torch.zeros_like(x)
    x_temp = F.pad(x.unsqueeze(1), (0, 0, 1, 1),
                               'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:, 2:]
    return output


def extract_lfcc(file_index, audio_path, num_files, num_frames, sub_path, audio_type, with_delta=True, use_torch=False):


    for i in tqdm(range(num_files)):

        audio_sr = 16000
        if audio_type == "opus":
            audio_sr = 48000
        audio, sr = librosa.load(os.path.join(audio_path, file_index[i] + '.' + audio_type), sr=audio_sr)

        max_length = num_frames * 160 + 160
        audiosize = audio.shape[0]

        if audiosize <= max_length and use_torch:
            shortage = max_length - audiosize
            audio = np.pad(audio, (0, shortage), "wrap")
            audiosize = max_length
        
        if use_torch:
            audio = signal.lfilter([1, -0.97], [1], audio)
            trans_lfcc = torchaudio.transforms.LFCC(sample_rate=16000, 
                                            n_filter=20, 
                                            n_lfcc=20, 
                                            speckwargs={'n_fft':512, 'win_length':320, 'hop_length':160, 'window_fn':torch.hamming_window}
            )
            x = torch.FloatTensor(audio)
            x = trans_lfcc(x)
            if with_delta:
                x = x.transpose(0,1)
                x = x.unsqueeze(0)
                lfcc_delta = delta(x)
                lfcc_delta_delta = delta(lfcc_delta)
                x = torch.cat((x, lfcc_delta, lfcc_delta_delta), 2)
                x = x.squeeze(0)
                x = x.transpose(0,1)
                
        else:
            audio = torch.Tensor(np.expand_dims(audio, axis=0))
            trans_lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
            x = trans_lfcc(audio)
            x = x.squeeze(0)
            x = x.transpose(0,1)

        torch.save(x, os.path.join(sub_path, file_index[i]+'.pt'))

def extract_cqt(file_index, audio_path, num_files, sub_path, audio_type):

    duration = 6.4
     
    for i in tqdm(range(num_files)):
        
        x, fs = librosa.load(os.path.join(audio_path, file_index[i] + '.' + audio_type), sr=16000)

        len_sample = int(duration * fs)
        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)

        x = x[0: int(len_sample - 256)]

        x = signal.lfilter([1, -0.97], [1], x)
        x_cqt = librosa.cqt(x, sr=fs, hop_length=256, n_bins=432, bins_per_octave=48, window='hann', fmin=15)
        pow_cqt = np.square(np.abs(x_cqt))
        log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)

        torch.save(log_pow_cqt, os.path.join(sub_path, file_index[i]+'.pt'))


def save_feature(protocol_path, data_path, feature_path, feature_type, status='train'):

    protocol = pd.read_csv(protocol_path, sep=' ', header=None).values
    if status=="eval2021" or status=="df2021":
        file_index = protocol[:,0]
    else:
        file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    num_frames = 798

    if status=="train":
        for i in range(len(aug_type)):
            sub_path = os.path.join(feature_path, aug_type[i], feature_type)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            audio_path = os.path.join(data_path, aug_type[i])
            console.print(audio_path, style="cyan")
            if feature_type == "lfcc":
                extract_lfcc(file_index, audio_path, num_files, num_frames, sub_path, file_type[i])
            if feature_type == "cqt":
                extract_cqt(file_index, audio_path, num_files, sub_path, file_type[i])
    else:
        sub_path = os.path.join(feature_path, "flac", feature_type)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        
        audio_path = os.path.join(data_path, aug_type[0])
        if feature_type=="lfcc":
            extract_lfcc(file_index, audio_path, num_files, num_frames, sub_path, file_type[0])
        if feature_type=="cqt":
            extract_cqt(file_index, audio_path, num_files, sub_path, file_type[0])
    console.print('LFCC features has been generated.', style="cyan")


if __name__ == '__main__':


    ##############################################################
    #### Just need to change the root_path and data type
    #### For preprocess to extract LFCC with torch or baseline
    ##############################################################

    ######################Options and database####################
    root_path = 'C:/Users/alex/diskdata/corpora/ASVspoof2019/LA'
    status = status_type[0]
    #############################END##############################
    data_path = os.path.join(root_path, "ASVspoof2019_LA_" + status)
    if status == "train":
        protocol_path = os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    else:
        protocol_path = os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.' + status +'.trl.txt')

    if status == "eval2021":
        protocol_path = os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt')
        data_path     = os.path.join(root_path, 'ASVspoof2021_LA_eval')

    if status == "df2021":
        protocol_path = os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt')
        data_path     = os.path.join(root_path, 'ASVspoof2021_DF_eval')
    

    # create folders for new types of data
    feature_path = os.path.join(root_path, 'feature_data', status)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    # generate cqt feature per sample
    console.print('Generating LFCC data...', style="green")
    save_feature(protocol_path, data_path, feature_path, "lfcc", status)
    console.print('End of Program.',style="red")
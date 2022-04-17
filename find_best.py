import glob
import os
import numpy as np
from tqdm import tqdm
from tools.dataset_loader import ASVDataset
from tools.evaluate import eval_to_score_file


file_root = "./scores/Res2block-down4-aug-MHA3-scorefiles"
score_files = glob.glob('%s/cm_score*.txt'%file_root)
score_files.sort()
EER =[] 
tDCF=[]

for file in tqdm(score_files):
	eer, min_tDCF = eval_to_score_file(file, phase="progress")
	EER.append(eer)
	tDCF.append(min_tDCF)

print("CM EER: {:.2f}".format(min(EER)), "min_tDCF: {:.4f}".format(tDCF[int(EER.index(min(EER)))]))
print(score_files[int(EER.index(min(EER)))])


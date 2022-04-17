import argparse
import os,glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.loss import OCSoftmax
from tools.dataset_loader import ASVDataset
from tools.tdnn import TDNN
from tools.evaluate import eval_to_score_file, dfeval_to_score_file


def test_model(feat_model_path, oc_model_path, device, batch_size, test_type, **kwargs):

    feat_model = TDNN(**vars(args)).to(device)
    ocsoftmax  = OCSoftmax(**vars(args)).to(device)
    feat_model.load_state_dict(torch.load(feat_model_path, map_location="cuda"))
    ocsoftmax.load_state_dict(torch.load(oc_model_path, map_location="cuda"))
    feat_model.eval()
    ocsoftmax.eval()

    if test_type == "eval2021":
        test_set_2021 = ASVDataset("eval2021-eval")
        test_dataloader_2021 = DataLoader(test_set_2021, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=3)
        score_file = './scores/cm_score_2021LA'+'-' + args.conv_way+'-'+args.pooling_way + '.txt'
        with open(score_file, 'w') as cm_score_file:
            for batch_x, batch_y, batch_meta in tqdm(test_dataloader_2021):
                batch_x = batch_x.to(device)
                labels = batch_y.to(device)

                if args.pooling_way == "MHA" or args.pooling_way == "MHA3":
                    feats, p  = feat_model(batch_x)
                else:
                    feats     = feat_model(batch_x)
                oc_loss, score = ocsoftmax(feats, labels, is_train=False)
                for j in range(labels.size(0)):
                    cm_score_file.write('%s %s %s\n' % (batch_meta.file_name[j], score[j].item(), 'bonafide' if labels[j] == float(1) else 'spoof'))

        cm_eer, min_tDCF = eval_to_score_file(os.path.join('', score_file), phase="eval")
        print("CM EER: {:.2f}".format(cm_eer), "min_tDCF: {:.4f}".format(min_tDCF))

    elif test_type == "df2021":

        test_set_df2021 = ASVDataset("df2021-eval")
        test_data_loader_df2021 = DataLoader(test_set_df2021, batch_size=batch_size, shuffle=False, num_workers=4,  pin_memory=True, prefetch_factor=3)
        score_file = './scores/cm_score_2021DF'+'-' + args.conv_way+'-'+args.pooling_way + '.txt'
        with open(score_file, 'w') as cm_score_file:
            for batch_x, batch_y, batch_meta in tqdm(test_data_loader_df2021):
                batch_x = batch_x.to(device)
                labels = batch_y.to(device)

                if args.pooling_way == "MHA" or args.pooling_way == "MHA3":
                    feats, p  = feat_model(batch_x)
                else:
                    feats     = feat_model(batch_x)
                oc_loss, score = ocsoftmax(feats, labels, is_train=False)

                for j in range(labels.size(0)):
                    cm_score_file.write('%s %s %s\n' % (batch_meta.file_name[j], score[j].item(), 'bonafide' if labels[j] == float(1) else 'spoof'))

        cm_eer = dfeval_to_score_file(os.path.join('', score_file), phase="eval")
        print("CM EER: {:.2f}".format(cm_eer*100))
    return


def test(device, batch_size, test_type, **kwargs):
    feat_model_name      = args.conv_way + "-" + args.pooling_way + "-" + "feat"
    oc_name         = args.conv_way + '-' + args.pooling_way + "-" + "oc"
    model_name = 'model_%03d.pt' % args.model_index
    feat_model_path = os.path.join("./models", feat_model_name, model_name)
    oc_model_path = os.path.join("./models", oc_name, model_name)
    print(test_model(feat_model_path, oc_model_path, **vars(args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # chose best epoch using find_best
    parser.add_argument('--model_index', type=int, help="The index of feat and oc model", default=59)
    parser.add_argument('--batch-size', type=int, help="batch size for test process", default=24)
    parser.add_argument("--test_type", type=str, help="The type of test set: eval, eval2021, df2021", default="eval2021")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument("--feature-type", type=str, help="type of feature", default="lfcc")
    parser.add_argument("--feature-dim", type=int, help="feature dimension", default=60)
    parser.add_argument("--device", type=str, help="Test use which type of deviece", default="cuda")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)
    parser.add_argument("--pooling_way", type=str, help="ways of pooling, ASP or MHA", default="ASP")
    parser.add_argument("--conv_way", type=str, help="Res2block or SCblock", default="Res2block")
    parser.add_argument('--channel', type=int, default=512, help="The number of the TDNN model's channel")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    test(**vars(args))

import torch
import argparse
import os
import time
import numpy as np
import sys
import glob

from torch.utils.data import DataLoader
from collections import defaultdict
from rich.console import Console
from tqdm import tqdm

from tools.dataset_loader import ASVDataset
from tools.evaluate import eval_to_score_file
from tools.resnet import setup_seed, Res2Net, ResNet
from tools.tdnn import TDNN
from tools.loss import *

console = Console()

def add_parser(parser):
    parser.add_argument("--feature_type", type=str, help="type of feature", default="LFCC")
    parser.add_argument("--feature_dim", type=int, help="feature dimension", default=60)
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)
    parser.add_argument("--device", type=str, help="trainning use which device", default="cuda")
    parser.add_argument("--pooling_way", type=str, help="ways of pooling, ASP or MHA or GMH", default="ASP")
    parser.add_argument("--conv_way", type=str, help="ways of conv, Res2block or SCblock", default="Res2block")
    parser.add_argument("--context", type=bool, help="whether use context", default=True)

    parser.add_argument('--channel', type=int, default=512, help="The number of the TDNN model's channel")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Mini batch size for validation")
    parser.add_argument('--test_step', type=int, default=1, help="how many epochs make test")
    parser.add_argument('--epoch', type=int, default=0, help="current epoch number")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr-decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=30, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--model_path', type=str, help="saved model path")
    parser.add_argument('--ocsoftmax', type=str, help="saved ocsoftmax path")
    parser.add_argument('--model_save_path', type=str, help="saved model path", default="./models")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    # assign device
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        console.print('Cuda device is available, your device is :', torch.cuda.get_device_name(0), style="bold magenta")
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(parser, device):

    console.print("Loading train dataset...", style="green", sep="")
    args = parser.parse_args()

    feat_model = TDNN(**vars(args)).to(device)

    console.print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in feat_model.parameters())), style="red")   

    feat_optimizer = torch.optim.Adam(
                                feat_model.parameters(), 
                                lr=args.lr,
                                betas=(args.beta_1, args.beta_2), 
                                eps=args.eps, 
                                weight_decay=0.0005
                                )

    ocsoftmax = OCSoftmax(**vars(args)).to(args.device)                       
    ocsoftmax_optimzer = torch.optim.SGD(
                        ocsoftmax.parameters(), 
                        lr=args.lr
                        )


    train_set = ASVDataset(type="train")
    test_set   = ASVDataset(type="eval2021-progress")

    train_loader = DataLoader(
                                train_set, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                shuffle=True, 
                                pin_memory=True,
                                prefetch_factor=3
                                )
    test_loader  = DataLoader(
                                test_set, 
                                batch_size=args.test_batch_size, 
                                num_workers=args.num_workers, 
                                shuffle=False, 
                                pin_memory=True,
                                prefetch_factor=3
                                )
    


    # Checkpoint
    if args.model_path and args.ocsoftmax:
        feat_model.load_state_dict(torch.load(args.model_path))
        ocsoftmax.load_state_dict(torch.load(args.ocsoftmax))
        console.print('Model loaded : {}'.format(args.model_path), style="red")

    model_name      = args.conv_way + "-" + args.pooling_way + "-" + "feat"
    oc_name         = args.conv_way + '-' + args.pooling_way + "-" + "oc"
    feat_modelfiles = glob.glob('%s/%s/model_*.pt'%(args.model_save_path,model_name))
    feat_modelfiles.sort()
    oc_modelfiles = glob.glob('%s/%s/model_*.pt'%(args.model_save_path,oc_name))
    oc_modelfiles.sort()
    if len(feat_modelfiles) >= 1:
        console.print("Model %s loaded from previous state!"%feat_modelfiles[-1], style='red')
        args.epoch = int(os.path.splitext(os.path.basename(feat_modelfiles[-1]))[0][6:])
        feat_model.load_state_dict(torch.load(feat_modelfiles[-1]))
        ocsoftmax.load_state_dict(torch.load(oc_modelfiles[-1]))

    # Training
    for epoch in range(args.epoch,  args.num_epochs):
        start = time.time()

        console.print('Training Epoch: ', epoch+1, style="blue")
        feat_model.train()
        ocsoftmax.train()
        train_loss_dict = defaultdict(list)
        dev_loss_dict = defaultdict(list)

        adjust_learning_rate(args, args.lr, feat_optimizer, epoch)
        adjust_learning_rate(args, args.lr, ocsoftmax_optimzer, epoch)

        num = 0
        sum_loss = 0
        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device,  non_blocking=True)
            labels = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

            if args.pooling_way == "MHA" or args.pooling_way == "MHA3":
                feats,p     = feat_model(batch_x)       
                oc_loss, score = ocsoftmax(feats, labels)
                oc_loss     = oc_loss + p
            else:
                feats = feat_model(batch_x)       
                oc_loss, score = ocsoftmax(feats, labels)


            feat_optimizer.zero_grad()
            ocsoftmax_optimzer.zero_grad()
            oc_loss.backward()
            feat_optimizer.step()
            ocsoftmax_optimzer.step()

            sum_loss += oc_loss.detach().cpu().numpy()
            train_loss_dict["loss"].append(oc_loss.item())

            with open(os.path.join('./log/', model_name + "-"+ 'train_loss.log'), 'a') as log:
                log.write(str(epoch+1) + "\t" +
                          str(np.nanmean(train_loss_dict["loss"])) + "\n")

            num = num + 1
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch+1, feat_optimizer.param_groups[0]["lr"], 100 * (num / train_loader.__len__())) + \
			" oc loss: %.5f \r"        %(sum_loss/(num)) )
            sys.stderr.flush()
        sys.stdout.write("\n")

        featmodel_save_path = os.path.join('./models/', '%s' % model_name)
        ocsoftmax_save_path = os.path.join('./models/', '%s' % oc_name)
        if os.path.exists(featmodel_save_path) is False:
            os.makedirs(featmodel_save_path)
        if os.path.exists(ocsoftmax_save_path) is False:
            os.makedirs(ocsoftmax_save_path)

        torch.save(feat_model.state_dict(), os.path.join(featmodel_save_path, 'model_%03d.pt' %(epoch + 1)))
        torch.save(ocsoftmax.state_dict(), os.path.join(ocsoftmax_save_path, 'model_%03d.pt' %(epoch + 1)))
        
        # test the model
        if (epoch+1) % args.test_step == 0:

            console.print("start test phase...", style="bold cyan")
            feat_model.eval()
            ocsoftmax.eval()
            score_file_name = args.conv_way + "-" + args.pooling_way + "-" + "scorefiles"
            score_save_path = os.path.join("./scores/", score_file_name)
            if os.path.exists(score_save_path) is False:
                os.makedirs(score_save_path)
            score_file = score_save_path + "/cm_score" + str(epoch+1) + ".txt"


            with torch.no_grad():
                with open(score_file, "w") as cm_score_file:
                    idx_loader, score_loader = [], []
                    for batch_x, batch_y, batch_meta in tqdm(test_loader):

                        batch_x = batch_x.to(device)
                        labels = batch_y.to(device)

                        if args.pooling_way == "MHA" or args.pooling_way == "MHA3":
                            feats, p = feat_model(batch_x)
                        else:
                            feats    = feat_model(batch_x)
                            
                        oc_loss, score = ocsoftmax(feats, labels, is_train=False)

                        dev_loss_dict['oc_loss'].append(oc_loss.item())
                        idx_loader.append(labels)
                        score_loader.append(score)

                        for j in range(labels.size(0)):
                            cm_score_file.write('%s %s %s\n' % (batch_meta.file_name[j], score[j].item(), 'bonafide' if labels[j] == float(1) else 'spoof'))

                cm_eer, min_tDCF = eval_to_score_file(os.path.join('', score_file), phase="progress")
                with open(os.path.join('./log/', model_name + "-"+ 'dev_loss.log'), "a") as log:
                    log.write(str(epoch+1) + "\t" + 
                              str(np.nanmean(dev_loss_dict["oc_loss"])) + "\t" +
                              str(cm_eer) + "\t" + str(min_tDCF) + "\n"
                    )
                console.print("CM EER: {:.2f}".format(cm_eer), "min_tDCF: {:.4f}".format(min_tDCF), style="yellow")

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        console.print("This epoch training time: {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)), style="green")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser('ASVSpoof2021')
    add_parser(parser)
    train(parser, device)


if __name__ == '__main__':
    main()

import os
import argparse
from torchvision import models
import torch
from torch import nn
from torch.autograd import Variable
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from random import choice
from PIL import Image
import cv2
import csv
from torchvision.utils import save_image
import nefro_dataset as nefro

# files_path = '/nas/softechict-nas-1/fpollastri/data/istologia/files//'
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--label', default='mesangiale', help='label to learn')
# parser.add_argument('--network', default='resnet101')
# parser.add_argument('--dropout', action='store_true', help='DropOut')
# parser.add_argument('--classes', type=int, default=1, help='number of epochs to train')
# parser.add_argument('--loadEpoch', type=int, default=0, help='load pretrained models')
# parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
# parser.add_argument('--batchSize', type=int, default=64, help='batch size during the training')
# parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
# parser.add_argument('--thresh', type=float, default=0.5, help='number of data loading workers')
# parser.add_argument('--epochs', type=int, default=41, help='number of epochs to train')
# parser.add_argument('--size', type=int, default=512, help='size of images')
# parser.add_argument('--n_forwards', type=int, default=1, help='number of different forwards to compute')
# parser.add_argument('--savemodel', type=int, default=10, help='number of epochs between saving models')
# parser.add_argument('--SRV', action='store_true', help='is training on remote server')
# parser.add_argument('--from_scratch', action='store_true', help='not finetuning')
#
# opt = parser.parse_args()
# print(opt)


class MyResnet(nn.Module):
    def __init__(self, net='resnet101', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyResnet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'resnet18':
            resnet = models.resnet18(pretrained)
            bl_exp = 1
        elif net == 'resnet34':
            resnet = models.resnet34(pretrained)
            bl_exp = 1
        elif net == 'resnet50':
            resnet = models.resnet50(pretrained)
            bl_exp = 4
        elif net == 'resnet101':
            resnet = models.resnet101(pretrained)
            bl_exp = 4
        elif net == 'resnet152':
            resnet = models.resnet152(pretrained)
            bl_exp = 4
        else:
            raise Warning("Wrong Net Name!!")
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AvgPool2d(int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(512 * bl_exp, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        x = self.last_fc(x)
        return x


class MyDensenet(nn.Module):
    def __init__(self, net='resnet101', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyDensenet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'densenet':
            densenet = models.densenet121(pretrained)
        else:
            raise Warning("Wrong Net Name!!")
        self.densenet = nn.Sequential(*(list(densenet.children())[0]))
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        x = self.last_fc(x)
        return x


class NefroNet:
    def __init__(self, net='resnet18', dropout=True, num_classes=2, size=512, lbl_name='mesangiale', device='cpu'):
        # Hyper-parameters
        self.net = net
        self.dropout = dropout
        self.num_classes = num_classes
        self.size = size
        self.lbl_name = lbl_name
        self.device = device
        self.models_dir = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/ckpts"
        if lbl_name == 'intensity':
            self.num_classes = 7

        self.nname = self.net + '_' + self.lbl_name
        if self.dropout:
            self.nname = 'dropout_' + self.nname

        if self.net == 'densenet':
            self.n = MyDensenet(net=self.net, pretrained=False, num_classes=self.num_classes,
                                dropout_flag=self.dropout).to(self.device)
        else:
            self.n = MyResnet(net=self.net, pretrained=False, num_classes=self.num_classes,
                              dropout_flag=self.dropout).to(self.device)

    def compute(self, x):
        with torch.no_grad():
            sigm = nn.Sigmoid()
            sofmx = nn.Softmax(dim=-1)
            self.n.eval()

            start_time = time.time()
            x = Image.open(x)
            # measure data loading time
            #print("data time: " + str(time.time() - start_time))

            preprocess = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                # QUA SI PUO' PROVARE A FARE IL CAMBIO USANDO LA MIA NORMALIZZAZIONE NEI VARI LIVELLI O QUELLA DI POLLASTRI 
                # Pollastri
                # transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))])
                # Lv1 Magistroni
                # transforms.Normalize((0.13496943, 0.14678506, 0.13129657),(0.19465959, 0.19976119, 0.19709547))])
                # Lv0 Magistroni, se io uso solo il canale verde e lo replico 3 volte, devo usare il valore di normalizzazione solo di quel canale 
                # transforms.Normalize((0.10321408, 0.1319403,  0.07907565), (0.16581197, 0.18537317, 0.16567207))
                # transforms.Normalize((0.10321408, 0.1319403,  0.07907565), (0.16581197, 0.18537317, 0.16567207))
                transforms.Normalize((0.1319403,0.1319403,0.1319403), (0.18537317,0.18537317,0.18537317))])
        
            x = preprocess(x)

##########################################################
            # # Compute output
            # x = x.unsqueeze(0).to(self.device)
            # output = torch.squeeze(self.n(x))
            # check_output = sofmx(output)
            # check_output, res = torch.max(check_output, -1)
            # res = res.item()
            # if self.num_classes == 2:
            #     res = bool(res)
            # else:
            #     res /= 2
            # # else:                 
            # #     check_output = sigm(output)
            # #     res = (check_output > 0.5)
            # # measure total time
            # #print("total time: " + str(time.time() - start_time))
            # return res
###########################################################

            # Compute riscritta

            x = x.unsqueeze(0).to(self.device)
            output = torch.squeeze(self.n(x))

            if self.num_classes == 1:
                check_output = torch.sigmoid(output)
                res = (check_output > 0.5).item()  # bool
            elif self.num_classes == 2:
                check_output = torch.softmax(output, dim=-1)
                _, res = torch.max(check_output, dim=-1)
                res = bool(res.item())
            else:  # num_classes > 2
                check_output = torch.softmax(output, dim=-1)
                _, res = torch.max(check_output, dim=-1)
                res = res.item()


    def find_stats(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _, _ in self.data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntraining")
        print("mean: " + str(mean) + " | std: " + str(std))

    def load(self):
        nname = self.net + '_' + self.lbl_name
        if self.dropout:
            nname = 'dropout_' + nname
        self.n.load_state_dict(torch.load(os.path.join(self.models_dir, nname + '_net.pth'), map_location=self.device))
        print("model weights successfully loaded")


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.moveaxis(np.float32(img), 0, -1)
    cam = cam / np.max(cam)
    cv2.imwrite('/homes/fpollastri/nefro_GradCam/' + name + '_cam.png', np.uint8(255 * cam))


def init_nets():
    lbl_d = {
        0: 'mesangial',
        1: 'parietal',
        2: 'cont. reg. parietal',
        3: 'irregular parietal',
        4: 'coarse granular',
        5: 'fine granular',
        6: 'segmental',
        7: 'global',
        # 8: 'intensity',
    }
    nets = []
    for lbl in lbl_d.values():
        n = NefroNet(lbl_name=lbl.split(' ')[0])
        n.load()
        nets.append(n)
    return nets

# if __name__ == '__main__':
# n = NefroNet(net=opt.network, dropout=opt.dropout, num_classes=opt.classes, num_epochs=opt.epochs,
#              size=opt.size,
#              batchSize=opt.batchSize, thresh=opt.thresh, pretrained=(not opt.from_scratch),
#              l_r=opt.learning_rate, n_workers=opt.workers, lbl_name=opt.label, write_flag=False)

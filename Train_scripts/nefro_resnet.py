import os
import csv
import cv2
import time
import json
from matplotlib.path import Path
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import argparse
import numpy as np
import imgaug as ia
from torch import nn
from collections import Counter
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import nefro_dataset as nefro_4k_and_diapo
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
from pathlib import Path as FilePath
import wandb
import os
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import numpy as np
import datetime
import ast
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# Aggiungi il path della cartella principale al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calibration.utils_calibration import (
    compute_brier, average_confidence_per_bin, accuracy_per_bin, compute_ECE
)
from calibration.grad_cam import GradCam, GuidedBackprop, save_gradient_images
from calibration.utils_nnets import categorical_to_one_hot
import Inference_scripts.nefro_dataset as nefro

#from Pollastri_Glomeruli.new_pg_GAN.progressBar import printProgressBar



class ConfusionMatrix:
    def __init__(self, num_classes):
        self.conf_matrix = np.zeros((num_classes, num_classes), int)

    def update_matrix(self, out, target):
        # I'm sure there is a better way to do this
        for j in range(len(target)):
            self.conf_matrix[out[j].item(), target[j].item()] += 1

    def get_metrics(self):
        samples_for_class = np.sum(self.conf_matrix, 0)
        diag = np.diagonal(self.conf_matrix)

        acc = np.sum(diag) / np.sum(samples_for_class)
        w_acc = np.divide(diag, samples_for_class)
        w_acc = np.mean(w_acc)

        return acc, w_acc


class MyResnet(nn.Module):

    def __init__(self, net='resnet101', pretrained=False, num_classes=1, dropout_flag=False, bl_exp=4):
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
        self.avgpool = nn.AvgPool2d(int(opt.size / 32), stride=1)
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
    def __init__(self, net='resnet101', pretrained=False, num_classes=1, dropout_flag=False):
        super(MyDensenet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'densenet':
            densenet = models.densenet121(pretrained)
        else:
            raise Warning("Wrong Net Name!!")
        self.densenet = nn.Sequential(*(list(densenet.children())[0]))
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=int(opt.size / 32), stride=1)
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


class ImgAugTransform:
    def __init__(self, config_code, size=512, SRV=False):
        self.SRV = SRV
        self.size = size
        self.config = config_code

        sometimes = lambda aug: ia.augmenters.Sometimes(0.5, aug)

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE
        cc = max(self.config, 0)
        if cc % 2:
            self.mode = 'reflect'
            cc -= 1
        else:
            self.mode = 'constant'

        self.possible_aug_list = [
            None,
            None,
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=False, name="PoissonNoise")),
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False, name="Dropout")),
            sometimes(ia.augmenters.GaussianBlur((0, 0.8), name="GaussianBlur")),
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10), name="HueSaturationShift")),
            sometimes(ia.augmenters.GammaContrast((0.5, 1.5), name="GammaContrast")),
            None,
            None,
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.04), name="PiecewiseAffine")),
            sometimes(ia.augmenters.Affine(shear=(-20, 20), mode=self.mode, name="ShearAffine")),
            sometimes(ia.augmenters.CropAndPad(percent=(-0.2, 0.05), pad_mode=self.mode, name="CropAndPad")),
        ]

        self.aug_list = [
            ia.augmenters.Fliplr(0.5, name="FlipLeftRight"),
            ia.augmenters.Flipud(0.5, name="FlipUpDown"),
            ia.augmenters.Affine(rotate=(-180, 180), mode=self.mode, name="Rotation"),
        ]

        for i in range(len(self.possible_aug_list)):
            if cc % 2:
                aug = self.possible_aug_list[i]
                if aug is not None:
                    self.aug_list.append(aug)
            cc = cc // 2
            if not cc:
                break

        # Dopo che ho aggiunto tutte le trasformazioni
        self.aug = ia.augmenters.Sequential(self.aug_list)

        if self.config >= 0:
            print(self.mode)
            for a in self.aug_list:
                print(a.name)

    def get_aug_transf(self):
        return self.aug_list

    def __call__(self, img):

        self.aug.reseed(random.randint(1, 10000))

        img = np.array(img, dtype='uint16')
        if not self.SRV:
            plot(img)
        img = ia.augmenters.PadToFixedSize(width=max(img.shape[0], img.shape[1]),
                                           height=max(img.shape[0], img.shape[1]),
                                           pad_mode=self.mode, position='center').augment_image(img)
        img = ia.augmenters.Resize({"width": self.size, "height": self.size}).augment_image(img)
        if self.config == -1:
            if not self.SRV:
                plot(img)
            return img.astype('int32')
        else:
            if not self.SRV:
                plot(self.aug.augment_image(img))
            return self.aug.augment_image(img).astype('int32')


class NefroNet():
    def __init__(self, wandb_flag, sampler, old_or_new_folder, project_name, wloss, net, dropout, num_classes, num_epochs, l_r, scheduler, size, batch_size, val_loss, n_workers, thresh, lbl_name, conf_matrix_lbl, w4k,
                 wdiapo, load_for_fine_tuning, weights_path, models_dir, augm_config=0, pretrained=True, write_flag=False):

        self.project_name = project_name

        # Hyper-parameters
        self.net = net
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.learning_rate = l_r
        self.size = size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.wloss = wloss
        self.thresh = thresh
        self.lbl_name = lbl_name
        self.conf_matrix_lbl = conf_matrix_lbl
        self.w4k = w4k
        self.wdiapo = wdiapo
        self.write_flag = write_flag
        self.models_dir = models_dir
        self.wandb_flag = wandb_flag
        self.sampler = sampler
        self.old_or_new_folder = old_or_new_folder
        self.load_for_fine_tuning = load_for_fine_tuning
        self.weights_path = weights_path
        self.val_loss = val_loss
        self.scheduler = scheduler
        self.pretrained = pretrained


        print("GPU disponibile:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Nome GPU:", torch.cuda.get_device_name(0))
            print("Memoria totale (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
            print('Sto trainando la classe: ', self.lbl_name)
            print('Il numero di classi è :', self.num_classes)
            print('La conf matrix ha label : ', self.conf_matrix_lbl)
            
       
        self.nname = self.net + '_' + str(self.lbl_name)
        if self.dropout:
            self.nname = 'dropout_' + self.nname

        # Di solito queste si mettono in una cartella che si chiama transforms
        imgaug_transforms = ImgAugTransform(config_code=augm_config, size=size, SRV=opt.SRV)
        inference_imgaug_transforms = ImgAugTransform(config_code=-1, size=size, SRV=opt.SRV)
        aug_trans = imgaug_transforms.get_aug_transf()

        aug_trans_list = []
        for i in range(len(aug_trans)):
            aug_trans_list.append(aug_trans[i].name) 

        self.aug_trans = aug_trans_list

        self.today = datetime.date.today()
        print(self.today) 

        # INIT WANDB
        if self.wandb_flag:
            wandb.init(
                project="Nefrologia",
                name=f"{self.project_name}_{self.lbl_name}_{self.today}",
                config={
                    "model": "ResNet18",
                    "num_classes": self.num_classes,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.num_epochs,
                    "thresh": self.thresh,
                    "label": self.lbl_name,
                    "img_aug": self.aug_trans,
                    "w4k": self.w4k,
                    "wdiapo": self.wdiapo,
                    "lr" : self.learning_rate
                }
            )

        
        dataset = nefro_4k_and_diapo.Nefro(
                                split='training',
                                old_or_new_folder = self.old_or_new_folder,
                                label_name=self.lbl_name,
                                w4k=self.w4k,
                                wdiapo=self.wdiapo,
                                size=(opt.size, opt.size),
                                transform=transforms.Compose([
                                    imgaug_transforms,  # questo deve avere __call__ definito
                                    # Questo non ci vuole perchè viene già fatto quando chiamo get_images
                                    #nefro.NefroTiffToTensor(),
                                    #transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))
                                ])
        )


        # Creo il weighted random sampler
        labels_list = dataset.labels_list
        class_distribution = dataset.class_distribution
        print('Questa è la distribuzione delle classi : ', class_distribution)
        total_samples = sum(class_distribution.values())
        class_weights = {cls: 1.0/count for cls, count in class_distribution.items()}
        print('Class_weights : ', class_weights)
        sample_weights = [class_weights[label] for label in labels_list]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        validation_dataset = nefro_4k_and_diapo.Nefro(split='validation', old_or_new_folder = self.old_or_new_folder, label_name=self.lbl_name, w4k=self.w4k,
                                                      wdiapo=self.wdiapo,
                                                      size=(opt.size, opt.size),
                                                      transform=transforms.Compose([
                                                        inference_imgaug_transforms,  # questo deve avere __call__ definito
                                                        #nefro.NefroTiffToTensor(),
                                                        #transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))
                                ]))


        eval_dataset_4k = nefro_4k_and_diapo.Nefro(split='test', old_or_new_folder = self.old_or_new_folder, label_name=self.lbl_name, w4k=True,
                                                   wdiapo=False,
                                                   size=(opt.size, opt.size),
                                                   transform=transforms.Compose([
                                                      inference_imgaug_transforms,  # questo deve avere __call__ definito
                                                      #nefro.NefroTiffToTensor(),
                                                      #transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))
                                ]))
        
        print('Il numero di dati nel dataset di train è :', len(dataset))
        print('Il numero di dati nel dataset di validation è :', len(validation_dataset))
        print('Il numero di dati nel test dataset è :', len(eval_dataset_4k))
       
        
        self.dataset_for_folds = nefro_4k_and_diapo.Nefro(
                                split='NewComplete',
                                old_or_new_folder = self.old_or_new_folder,
                                label_name=self.lbl_name,
                                w4k=self.w4k,
                                wdiapo=self.wdiapo,
                                size=(opt.size, opt.size),
                                transform=transforms.Compose([
                                    imgaug_transforms,  # questo deve avere __call__ definito
                                    # Questo non ci vuole perchè viene già fatto quando chiamo get_images
                                    #nefro.NefroTiffToTensor(),
                                    #transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))
                                ])
        )

        self.folds = self.kfold_train_val_test_split(self.dataset_for_folds, 4)
        self.check_no_overlap(self.folds)

        if self.net == 'densenet':
            self.n = MyDensenet(net=self.net, pretrained=pretrained, num_classes=self.num_classes,
                                dropout_flag=self.dropout).to('cuda')
        else:
            # Con sampler
            if self.sampler == 'True':
                print('Weighted sampler True')
                
                self.data_loader = DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            sampler=sampler,
                                            num_workers=self.n_workers,
                                            drop_last=True,
                                            pin_memory=True) # prefatch factor

            
            else:
                print('Weighted sampler False')
                self.data_loader = DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=self.n_workers,
                                            drop_last=True,
                                            pin_memory=True) # prefatch factor

            self.validation_data_loader = DataLoader(validation_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.n_workers,
                                                    drop_last=False,
                                                    pin_memory=True)

           
            # self.eval_data_loader_diapo = DataLoader(eval_dataset_diapo,
            #                                      batch_size=self.batch_size,
            #                                      shuffle=False,
            #                                      num_workers=self.n_workers,
            #                                      drop_last=False,
            #                                      pin_memory=True)


            self.eval_data_loader_4k = DataLoader(eval_dataset_4k,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.n_workers,
                                                  drop_last=False,
                                                  pin_memory=True)
            
        print('Numero dati di train : ', len(self.data_loader))
        print('Numero dati di validation : ', len(self.validation_data_loader))
        print('Numero dati di test : ', len(self.eval_data_loader_4k))
      

        self.n = self.init_model()

        # Loss and optimizer
        if self.num_classes == 1:
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Il parametro pesa solo i positivi quindi devo dimezzare il peso se ho il doppio dei positivi rispetto ai negativi
            # pos_weight = torch.tensor([3.8], device=device)
            # self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.criterion = nn.BCEWithLogitsLoss()
            # Weighted_random_sampler al posto di BCE
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.n.parameters()),
                                             lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', verbose=True,
                                                                        threshold=0.004)
        else:
            
            # Nel caso nel dataset non ci fossero esempi di classe 1 o 0 devo usare un epsilon per evitare la divisone per 0
            c0_w, c1_w = self.get_probabilities(self.data_loader)
           
            if self.lbl_name != [['INTENS']] and self.wloss=='True': # loss mi dice se voglio pesare o no la loss function 
                class_w = torch.tensor([c0_w, c1_w], device='cuda')
                print(f'La Loss è stata pesata con pesi {class_w} (num_classes = 2)')
                self.criterion = nn.CrossEntropyLoss(weight=class_w)
            else: 
                print('La loss non è stata pesata(num_classes = 2)')
                self.criterion = nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.n.parameters()),
                                               lr=self.learning_rate)

            
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.n.parameters()),
                                             lr=self.learning_rate)
            
            if self.scheduler != 'OneCycle':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', verbose=True,
                                                                            threshold=0.004)
            else:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.learning_rate,
                    steps_per_epoch=len(self.data_loader),
                    epochs=self.num_epochs,
                    pct_start=0.3
                )

            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        print("Scheduler : OneCycleLR")
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        print("Scheduler : ReduceLROnPlateau")
            else:
                        print("Altro scheduler")


    def extract_base_id(self, wsi_name):
        name = wsi_name.strip()

        if name.startswith("R24") or name.startswith("R23"):
            parts = name.replace(' ', '_').split('_')
            if len(parts) >= 3:
                return '_'.join(parts[:3])
            else:
                return name

        elif name.startswith("R22"):
            parts = name.split(' ')
            return parts[0] if parts else name

        else:
            return name

    def init_model(self):
        model = MyResnet(net=self.net, pretrained=True, num_classes=self.num_classes,
                        dropout_flag=self.dropout).to('cuda')
        if self.load_for_fine_tuning == 'True':
            print('Esperimento fine tuning! ...')
            checkpoint = torch.load(self.weights_path, map_location='cuda')
            try:
                print("Prima del caricamento:", model.last_fc.weight.data.norm())
                model.load_state_dict(checkpoint)
                print("Dopo il caricamento:", model.last_fc.weight.data.norm())
                print(f'Ho caricato i pesi del modello trovati nel percorso {self.weights_path}')
            except RuntimeError as error:
                print(f"Errore nel caricamento dei pesi {self.weights_path}")
        return model


    def check_no_overlap(self, results):
        tests = [set(test.tolist()) for _, _, test in results]
        for i in range(len(tests)):
            for j in range(i+1, len(tests)):
                inter = tests[i].intersection(tests[j])
                if inter:
                    print(f"I fold {i+1} e {j+1} condividono {len(inter)} elementi nei test set!")
                    return False
        print("Test set di tutti i fold sono disgiunti (nessuna sovrapposizione)")
        return True

    def group_indices_by_base_id(self, dataset):
        base_id_to_indices = defaultdict(list)

        for idx, img_path in enumerate(dataset.names):
            wsi_name = os.path.basename(os.path.dirname(img_path))  # cartella contenente l’immagine
            base_id = self.extract_base_id(wsi_name)
            base_id_to_indices[base_id].append(idx)

        return base_id_to_indices

    def kfold_train_val_test_split(self, dataset, k=4, seed=42):
        random.seed(seed)
        base_id_to_indices = self.group_indices_by_base_id(dataset)

        base_ids = list(base_id_to_indices.keys())
        random.shuffle(base_ids)

        # Distribuzione round-robin per assicurare che tutti i base_id siano inclusi
        folds = [[] for _ in range(k)]
        for i, bid in enumerate(base_ids):
            folds[i % k].append(bid)

        used_ids = set()
        for f in folds:
            used_ids.update(f)

        all_ids = set(base_ids)
        excluded_ids = all_ids - used_ids

        print(f"[DEBUG] Base ID totali: {len(all_ids)} | Usati nei fold: {len(used_ids)} | Esclusi: {excluded_ids}")

        
        results = []
        csv_rows = []
        for i in range(k):
            test_ids = folds[i]
            val_ids = folds[(i + 1) % k]  
            train_ids = [bid for j, f in enumerate(folds) if j not in (i, (i + 1) % k) for bid in f]
            
            test_indices = [idx for bid in test_ids for idx in base_id_to_indices[bid]]
            val_indices = [idx for bid in val_ids for idx in base_id_to_indices[bid]]
            train_indices = [idx for bid in train_ids for idx in base_id_to_indices[bid]]

            for idx in train_indices:
                img_path = dataset.names[idx]
                csv_rows.append((os.path.basename(img_path), i, "train"))
            for idx in val_indices:
                img_path = dataset.names[idx]
                csv_rows.append((os.path.basename(img_path), i, "val"))
            for idx in test_indices:
                img_path = dataset.names[idx]
                csv_rows.append((os.path.basename(img_path), i, "test"))

            results.append((
                np.array(train_indices),
                np.array(val_indices),
                np.array(test_indices)
            ))

        # Salva il CSV
        save_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Base_split_over_wsi/Cross_fold/folds.csv'
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "fold", "split"])
            writer.writerows(csv_rows)

        print(f"[INFO] File CSV salvato in: {save_path}")

        return results
        
    def freeze_layers(self, freeze_flag=True, nl=0):
        if nl:
            l = list(self.n.resnet.named_children())[:-nl]
        else:
            l = list(self.n.resnet.named_children())
        # list(list(self.n.resnet.named_children())[0][1].parameters())[0].requires_grad
        for name, child in l:
            for param in child.parameters():
                param.requires_grad = not freeze_flag

    def train(self):

        # Controllo monotonia step, devono crescere e non risettarsi a 1, ma dà un WARNING che non sono crescenti
        # Così loggo con step=epoch
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="epoch")
        wandb.define_metric("val/loss", step_metric="epoch")
        wandb.define_metric("val/accuracy", step_metric="epoch")
        wandb.define_metric("val/precision", step_metric="epoch")
        wandb.define_metric("val/recall", step_metric="epoch")
        wandb.define_metric("val/f1_score", step_metric="epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")

        for epoch in range(self.num_epochs):
            self.n.train()
            losses = []
            start_time = time.time()

            # PER DEBUG
            # image, label, name = self.data_loader.dataset[42]  # indice casuale
            # print(f"Nome immagine: {name}")
            # print(f"Etichetta: {label}")
            # print(f"Tipo: {type(image)}")
            # print(f"Shape immagine: {image.shape}")  # per immagini torch: (C, H, W)
            # print('_'*30)
            # image, label, name = self.data_loader.dataset[30]  # indice casuale
            # print(f"Nome immagine: {name}")
            # print(f"Etichetta: {label}")
            # print(f"Tipo: {type(image)}")
            # print(f"Shape immagine: {image.shape}")  # per immagini torch: (C, H, W)

            for i, (x, target, _) in enumerate(self.data_loader):

                # Stampiamo la distribuzione delle classi nel batch
                if i % 10 == 0:  
                    counts = Counter(target.tolist())
                    print(f"[Epoch {epoch} | Batch {i}] Class distribution in batch: {dict(counts)}")


                x = x.to('cuda')
                if self.num_classes == 1:
                    target = target.to('cuda', torch.float)
                    # Questo peso qui essendo fisso sarebbe meglio impostarlo al di fuori del training loop quando definisco la loss la prima volta, ma non dovrebbe dare errori
                    # Non capisco pero perchè non rifletta la distribuzione dei dati di training 
                    #self.criterion.weight = get_weights(target)
                    output = torch.squeeze(self.n(x))
                else:
                    target = target.to('cuda', torch.long)
                    output = self.n(x)
                loss = self.criterion(output, target)
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                lr = self.optimizer.param_groups[0]['lr']  # Get the current learning rate
                if self.scheduler == 'OneCycle':
                    self.scheduler.step()

            print(f'Epoch: {epoch} | loss: {np.mean(losses):.6f} | time: {time.time() - start_time:.2f}s')
            print('Validation: ')
            # Non sto facendo cross-validation quindi fold None
            fold = None
            val_acc, val_pr, val_recall, val_f_score, cm, _, _ = self.eval(self.validation_data_loader, epoch=epoch, fold=fold, write_flag=True)
            if self.scheduler != 'OneCycle':
                self.scheduler.step(val_f_score)

            if epoch % 5 == 0:
                self.thresh_eval(epoch)

            mean_loss = np.mean(losses)
            print(f'Questa è la media delle loss: {mean_loss:.6f}')

            # Trasformo tutto in float python per sicurezza (non tensori o None), perchè qua da problemi di log 
            try:
                val_acc = float(val_acc)
                val_pr = float(val_pr)
                val_recall = float(val_recall)
                val_f_score = float(val_f_score)
                mean_loss = float(mean_loss)
            except Exception as e:
                print(f"Errore nella conversione a float: {e}")
                continue  # salta logging per questo epoch se conversione fallisce

            try :
                wandb.log({"train/loss" : mean_loss, 'epoch': epoch})
                wandb.log({"val/accuracy" : val_acc, 'epoch': epoch})
                wandb.log({"val/precision" : val_pr, 'epoch': epoch})
                wandb.log({"val/recall" : val_recall, 'epoch': epoch})
                wandb.log({"val/f1_score" : val_f_score, 'epoch': epoch})
                wandb.log({"learning_rate": lr, 'epoch': epoch})  
            except Exception as e:
                print(f'Errore nel log di wandb nel train: {e}')
            
        wandb.finish()


    def train_test_on_folds(self):
  
        for fold, (train_indices, val_indices, test_indices) in enumerate(self.folds):

            wandb.init(
                project="Nefrologia Cross_Fold", 
                name=f"{self.project_name}_{self.lbl_name}_{self.today}_fold_{fold+1}",
                group=self.project_name,
                config={
                    "model": "ResNet18",
                    "num_classes": self.num_classes,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.num_epochs,
                    "thresh": self.thresh,
                    "label": self.lbl_name,
                    "img_aug": self.aug_trans,
                    "w4k": self.w4k,
                    "wdiapo": self.wdiapo,
                },
                reinit=True
            )

            wandb.define_metric("epoch")
            wandb.define_metric("train/loss", step_metric="epoch")
            wandb.define_metric("val/accuracy", step_metric="epoch")
            wandb.define_metric("val/precision", step_metric="epoch")
            wandb.define_metric("val/recall", step_metric="epoch")
            wandb.define_metric("val/f1_score", step_metric="epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")

            # Dataset e dataloader per fold
            train_subset = Subset(self.dataset_for_folds, train_indices)
            val_subset = Subset(self.dataset_for_folds, val_indices)
            self.train_fold_data_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.n_workers, drop_last=True, pin_memory=True)
            self.validation_fold_data_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.n_workers, drop_last=False, pin_memory=True)
       
            print(f'\n Fold {fold+1}/{len(self.folds)}')
            print(f'Train size: {len(self.train_fold_data_loader)}, Val size: {len(self.validation_fold_data_loader)}')

            c0_w, c1_w = self.get_probabilities(self.train_fold_data_loader)

            if self.lbl_name != [['INTENS']] and self.wloss:
                class_w = torch.tensor([c0_w, c1_w], device='cuda')
                print(f'Loss pesata con: {class_w}')
                self.criterion = nn.CrossEntropyLoss(weight=class_w)
            else:
                print('Loss NON pesata')
                self.criterion = nn.CrossEntropyLoss()

            # Inizializzo il modello per ogni fold altrimenti riparte dal training del fold precednete senza resettarsi
            self.n = self.init_model()

            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.n.parameters()), lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', verbose=True, threshold=0.004)

            # Ogni fold deve avere la sua best accuracy indipendente
            if self.lbl_name != [['INTENS']]:
                self.best_acc = 0.0
            else:
                self.best_acc = float('inf')
                
            for epoch in range(self.num_epochs):
                self.n.train()
                losses = []
                start_time = time.time()

                for i, (x, target, _) in enumerate(self.train_fold_data_loader):
                    if i % 10 == 0:
                        counts = Counter(target.tolist())
                        print(f"[Epoch {epoch} | Batch {i}] Class dist: {dict(counts)}")

                    x = x.to('cuda')
                    target = target.to('cuda', torch.float if self.num_classes == 1 else torch.long)
                    output = torch.squeeze(self.n(x)) if self.num_classes == 1 else self.n(x)

                    if self.lbl_name != [['INTENS']]:
                        loss = self.criterion(output, target)
                    elif self.lbl_name == [['INTENS']]:
                        intensity_prob = F.softmax(output, dim=1)
                        intensity_values = torch.arange(7).float().cuda()
                        intensity_values /= 2
                        intensity_expect = torch.sum(intensity_values * intensity_prob, 1)
                        loss = F.smooth_l1_loss(intensity_expect, target.float())

                    losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    lr = self.optimizer.param_groups[0]['lr']
                    if self.scheduler == 'OneCycle':
                        self.scheduler.step()

                print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f} | Time: {time.time() - start_time:.2f}s")
                print("Validation:")

                if self.lbl_name != [['INTENS']]:
                    val_acc, val_pr, val_recall, val_f_score, cm, _, _ = self.eval(self.validation_fold_data_loader, epoch, fold, write_flag=True)
                    # Logging su wandb
                    try:
                        wandb.log({
                            "train/loss": float(np.mean(losses)),
                            "val/accuracy": float(val_acc),
                            "val/precision": float(val_pr),
                            "val/recall": float(val_recall),
                            "val/f1_score": float(val_f_score),
                            "learning_rate": lr,
                            "epoch": epoch
                        })
                    except Exception as e:
                        print(f"[WANDB ERROR] {e}")

            

                elif self.lbl_name == [['INTENS']]:
                    acc, _ = self.eval_intensity(self.validation_fold_data_loader, epoch, fold, write_flag=True)

                if self.scheduler != 'OneCycle' and self.lbl_name != [['INTENS']]:
                    self.scheduler.step(val_f_score)
                else:
                    self.scheduler.step(acc)
                
            wandb.finish()
    
        print("\n Fine training inizio valutazione su test set per ciascun fold")
        result_path = f"/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Results/result_FoldSeed42_{self.lbl_name}.json"
        all_results = []

        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                try:
                    data = json.load(f)
                    all_results = data if isinstance(data, list) else [data]
                except json.JSONDecodeError:
                    all_results = []

        for fold, (_, _, test_indices) in enumerate(self.folds):
            print(f"\n Fold {fold} valutazione su test set")

            model_dir = os.path.join(self.models_dir, f"fold_{fold}")
            self.best_acc = 0  

            if self.old_or_new_folder == 'Files_old_Pollo/':
                if self.dropout:
                    model_name = f"dropout_{self.net}_{self.lbl_name}_Old_{fold}_net.pth"
                else:
                    model_name = f"{self.net}_{self.lbl_name}_Old_{fold}_net.pth"
            else:
                if self.dropout:
                    model_name = f"dropout_{self.net}_{self.lbl_name}_New_{fold}_net.pth"
                else:
                    model_name = f"{self.net}_{self.lbl_name}_New_{fold}_net.pth"

            model_path = os.path.join(model_dir, model_name)

            if not os.path.isfile(model_path):
                print(f"Modello non trovato per il fold {fold}: {model_path}")
                continue

            self.n.load_state_dict(torch.load(model_path))
            self.n.to("cuda")
            self.n.eval()

            test_subset = Subset(self.dataset_for_folds, test_indices)
            self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.n_workers, drop_last=False, pin_memory=True)

            if self.lbl_name != [['INTENS']]:
                acc, pr, rec, f1, cm, _, _ = self.eval(self.test_loader, epoch="final", fold=fold, write_flag=False)
                cm_pretty = f"""[[TN={cm[0,0]} FP={cm[0,1]}]
                                [FN={cm[1,0]} TP={cm[1,1]}]]"""

                res_dict = {
                    "Fold": fold,
                    "Commento": "Eval finale su test set dopo training su tutti i fold",
                    "Esperimento": vars(self.opt) if hasattr(self, "opt") else {},
                    "Accuracy": float(acc),
                    "Precision": float(pr),
                    "Recall": float(rec),
                    "Fscore": float(f1),
                    "Conf_matrix": cm_pretty,
                    "Weights": model_path
                }

                all_results.append(res_dict)

            elif self.lbl_name == [['INTENS']]:
                acc, conf_matrix, _, _ = self.eval_intensity(self.test_loader, epoch='final', fold=fold, write_flag=False )
               
                labels = ['0', '0.5', '1', '1.5', '2', '2.5', '3']
                cm_array = conf_matrix.conf_matrix

                header = "       " + " ".join([f"{label:>5}" for label in labels]) + "\n"
                rows = ""
                for i, row in enumerate(cm_array):
                    row_str = " ".join([f"{val:5d}" for val in row])
                    rows += f"{labels[i]:>5} | {row_str}\n"

                cm_pretty = "Confusion Matrix:\n" + header + rows

                res_dict = {
                    "Fold": fold,
                    "Commento": "Eval finale su test set dopo training su tutti i fold",
                    "Esperimento": vars(self.opt) if hasattr(self, "opt") else {},
                    "Accuracy": float(acc),
                    "Conf_matrix": cm_pretty,
                    "Weights": model_path
                }
                

        with open(result_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\nTutti i risultati salvati in: {result_path}")

    def write_on_descriptive_file(self, result_path, fold, images, y_pred_all):
        """
        Salvo i risultati della evaluation separata in un result files dove tengo traccia del nome del glomerulo e della sua prediction associata.
        """
        os.makedirs(result_path, exist_ok = True)   
        final_pth = os.path.join(result_path, f'fold{fold}.csv')
        with open(final_pth, 'a', newline='') as csvfile:
                    label_name = self.lbl_name[0] if isinstance(self.lbl_name, list) and len(self.lbl_name) > 0 else 'Prediction'
                    fieldnames = ['Glom_name', f'Prediction_{label_name}']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for img, pred in zip(images, y_pred_all):
                        writer.writerow({'Glom_name': img.split('/')[-1], f'Prediction_{label_name}': pred})
                
        print('Ho finito di scrivere il csv dei risultati')

    def separeted_evaluation_on_folds(self, folds_weights_pth, result_path):

        print("\n Inizio valutazione separata su test set per ciascun fold")

        for fold, (_, _, test_indices) in enumerate(self.folds):
            print(f"\n Fold {fold} valutazione su test set")

            model_dir = os.path.join(folds_weights_pth, str(self.lbl_name), f"fold_{fold}")

            self.best_acc = 0  

            if self.old_or_new_folder == 'Files_old_Pollo/':
                if self.dropout:
                    model_name = f"dropout_{self.net}_{self.lbl_name}_Old_{fold}_net.pth"
                else:
                    model_name = f"{self.net}_{self.lbl_name}_Old_{fold}_net.pth"
            else:
                if self.dropout:
                    model_name = f"dropout_{self.net}_{self.lbl_name}_New_{fold}_net.pth"
                else:
                    model_name = f"{self.net}_{self.lbl_name}_New_{fold}_net.pth"

            model_path = os.path.join(model_dir, model_name)

            if not os.path.isfile(model_path):
                print(f"Modello non trovato per il fold {fold}: {model_path}")
                continue

            self.n.load_state_dict(torch.load(model_path))
            self.n.to("cuda")
            self.n.eval()

            test_subset = Subset(self.dataset_for_folds, test_indices)
            self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.n_workers, drop_last=False, pin_memory=True)

            if self.lbl_name != [['INTENS']]:

                _, _, _, _, _, y_pred_all, images = self.eval(self.test_loader, epoch="final", fold=fold, write_flag=False)
                self.write_on_descriptive_file(result_path, fold, images, y_pred_all)
               
   
            elif self.lbl_name == [['INTENS']]:

                _, _, y_pred_all, images = self.eval_intensity(self.test_loader, epoch='final', fold=fold, write_flag=False)
                self.write_on_descriptive_file(result_path, fold, images, y_pred_all)
               
                
    # I livelli di intensità sono livelli da 0 a 3 con step 0.5.
    def train_intensity(self):
    
        self.best_acc = 5
        for epoch in range(self.num_epochs):
            self.n.train()
            losses = []
            start_time = time.time()
            for i, (x, target, _) in enumerate(self.data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to('cuda')
                target = target.to('cuda')
                #print(f"Target values unique: {torch.unique(target.cpu())}")

                # if self.num_classes == 1:
                #     target = target.to('cuda', torch.float)
                #     self.criterion.weight = get_weights(target)
                # else:
                #     target = target.to('cuda', torch.long)

                output = torch.squeeze(self.n(x))
                intensity_prob = F.softmax(output, dim=1)
                intensity_values = torch.arange(7).float().cuda()
                intensity_values /= 2
                intensity_expect = torch.sum(intensity_values * intensity_prob, 1)
                loss = F.smooth_l1_loss(intensity_expect, target.float())

                losses.append(loss.item())
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # # measure elapsed time
                # printProgressBar(i + 1, total + 1,
                #                  length=20,
                #                  prefix=f'Epoch {epoch} ',
                #                  suffix=f', loss: {loss.item()res = (check_output > 0.5).float() * 1:.3f}'
                #                  )

            print('Epoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)) + ' | time: ' + str(
                time.time() - start_time))
            print('Validation: ', end=' --> ')
            acc, _, _, _ = self.eval_intensity(self.validation_data_loader, epoch=epoch, write_flag=True)
            self.scheduler.step(acc)

            if epoch % 10 == 0:
                print('Facciamo una verifica sul test_set: ')
                print('Dataset 4k: ')
                self.eval_intensity(self.eval_data_loader_4k, epoch=epoch)
                # print('diapo: ')
                # self.eval_intensity(self.eval_data_loader_diapo, epoch=epoch)


    def eval(self, d_loader, epoch, fold, write_flag=False):
        val_losses = []
        y_true_all = []
        y_pred_all = []
        y_scores_all = []
        images = []

        with torch.no_grad():
            sigm = nn.Sigmoid()
            sofmx = nn.Softmax(dim=1)
            trues = 0
            g_trues = 0
            tr_trues = 0
            acc = 0
            self.n.eval()
            start_time = time.time()

            for i, (x, target, img_name) in enumerate(d_loader):
                x = x.to('cuda')
                output = torch.squeeze(self.n(x))

                if self.num_classes == 1:
                    target = target.to('cuda', torch.float)
                    check_output = sigm(output)
                    res = (check_output > self.thresh).float()
                    y_scores_all.extend(check_output.cpu().numpy().tolist())
                else:
                    target = target.to('cuda', torch.long)
                    check_output = sofmx(output)
                    check_output, res = torch.max(check_output, 1)
                    # EVAL LOSS
                    if self.val_loss == 'True':
                        loss = self.criterion(output, target)
                        val_losses.append(loss.item())

                    #y_scores_all.extend(check_output[:, 1].cpu().numpy().tolist())
                    #res = (check_output[:, 1] > self.thresh).int()

                # tr_target = target * s2 - 1  # only for custom F1 computation, fa riportare tr_target in dominio {-1, +1}
                tr_target = target

                # PER DEBUG
                # print(f"res unique values: {torch.unique(res)}")
                # print(f"tr_target unique values: {torch.unique(tr_target)}")

                tr_trues += (res == tr_target).sum().item()
                trues += res.sum().item()
                g_trues += target.sum().item()
                acc += (res.int() == target.int()).sum().item()

                # Stampa dei valori di accumulo per batch per capire meglio 
                #print(f"Batch {i}: tr_trues={tr_trues}, trues={trues}, g_trues={g_trues}")

                # Wandb rompe le palle se non metto tutta sta roba .cpu().int().numpy().tolist()
                y_true_all.extend(target.cpu().int().numpy().tolist())
                y_pred_all.extend(res.cpu().int().numpy().tolist())
                images.extend(img_name)

            # EVAL LOSS
            if self.val_loss == 'True':
                mean_loss = np.mean(val_losses)
                if wandb.run: 
                    wandb.log({"val/loss" : mean_loss, 'epoch': epoch})
                print(f"Validation Loss: {mean_loss:.6f}")


            class_names = self.conf_matrix_lbl
            print("Predizione 0 significa:", class_names[0])
            print("Predizione 1 significa:", class_names[1])  # e.g., ['Negative', 'Positive']

            y_pred_all = np.array(y_pred_all)
            y_true_all = np.array(y_true_all)
            print("\n Classification Report:")
            print(classification_report(
                y_true_all, 
                y_pred_all, 
                target_names=class_names,  # es. ['Globale', 'Segmentale']
                digits=4
            ))

            print("Etichette presenti nel validation set:", sorted(set(y_true_all)))

            # Calcolo TP, FP, FN su tutto il validation set
            tp = int(((y_pred_all == 1) & (y_true_all == 1)).sum())
            fp = int(((y_pred_all == 1) & (y_true_all == 0)).sum())
            fn = int(((y_pred_all == 0) & (y_true_all == 1)).sum())
            rec = tp / (tp + fn + 1e-5)
            predicted_positives = tp + fp
            pr = tp / (predicted_positives + 1e-5)
            cm = confusion_matrix(y_true_all, y_pred_all)
            print('Questa è la confusion matrix : ', cm)

            #pr = tr_trues / (trues + 1e-5)
            #rec = tr_trues / (g_trues + 1e-5)

            fscore = (2 * pr * rec) / (pr + rec + 1e-5)
            accuracy = acc / len(d_loader.dataset)

            stats_string = (
                f"Acc: {accuracy:.4f} | F1 Score: {fscore:.4f} | "
                f"Precision: {pr:.4f} | Recall: {rec:.4f} | "
                f"Ground Truth Trues: {g_trues} | Time: {time.time() - start_time:.2f}s"
            )
            print(stats_string)

            # PER Debugging
            # print("SAVING MODEL")
            # saved = self.save()
            # if saved:
            #     self.best_acc = accuracy

            if accuracy > self.best_acc and write_flag and epoch > 15:

                print(f"L'accuracy è {accuracy:.4f} mentre la best_accuracy è {self.best_acc:.4f}, quindi salvo i pesi")
                print("SAVING MODEL")
                saved = self.save(fold)
                if saved:
                    self.best_acc = accuracy

                if epoch % 5 != 0:
                    self.thresh_eval(epoch)

                try:
                    print('Logging confusion matrix')
                    wandb.log({
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=y_true_all,
                            preds=y_pred_all,
                            class_names=class_names
                        ),
                        "epoch": epoch
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to log confusion matrix: {e}")


        return accuracy, pr, rec, fscore, cm, y_pred_all, images


    def eval_intensity(self, d_loader, epoch, fold, write_flag=False):
        print('Eval Intensity...')
        conf_matrix = ConfusionMatrix(self.num_classes)
        with torch.no_grad():
            mae = 0
            mse = 0
            t_mae = 0
            t_mse = 0
            y_pred_all = []
            images = []
            self.n.eval()
            start_time = time.time()
            for i, (x, target, img_name) in enumerate(d_loader):
    
                images.extend(img_name)
                # compute output
                x = x.to('cuda')
                target = target.to('cuda')

                output = torch.squeeze(self.n(x))
                intensity_prob = F.softmax(output, dim=1)
                intensity_values = torch.arange(7).float().cuda()
                intensity_values /= 2
                intensity_expect = torch.sum(intensity_values * intensity_prob, 1)
                mae += F.l1_loss(intensity_expect, target.float(), reduction='sum').item()
                mse += F.mse_loss(intensity_expect, target.float(), reduction='sum').item()
                t_intensity_expect = torch.round(intensity_expect * 2) / 2
                t_mae += F.l1_loss(t_intensity_expect, target.float(), reduction='sum').item()
                t_mse += F.mse_loss(t_intensity_expect, target.float(), reduction='sum').item()
                conf_matrix.update_matrix((t_intensity_expect * 2).long(), (target * 2).long())
                
                float_preds = [val.item() for val in t_intensity_expect]
                y_pred_all.extend(float_preds)
                # Per debug
                # print("Predicted class counts:", torch.bincount((t_intensity_expect * 2).int()))
                # print("True class counts:", torch.bincount((target * 2).int()))

            stats_string = 'MAE: ' + str(mae / len(d_loader.dataset)) + ' | MSE: ' + str(mse / len(d_loader.dataset)) + \
                           ' | THRESHOLD MAE: ' + str(t_mae / len(d_loader.dataset)) + \
                           ' | THRESHOLD MSE: ' + str(t_mse / len(d_loader.dataset)) + \
                           ' | time: ' + str(time.time() - start_time)
            
            print(stats_string)

            if write_flag == True:
                wandb.log({
                    'MAE': mae / len(d_loader.dataset),
                    'MSE': mse / len(d_loader.dataset),
                    'THRESHOLD_MAE': t_mae / len(d_loader.dataset),
                    'THRESHOLD_MSE': t_mse / len(d_loader.dataset),
                    'Time': time.time() - start_time
                }, step=epoch)  # se epoch è disponibile

                                
            if (mae / len(d_loader.dataset)) < self.best_acc and write_flag:
                self.best_acc = mae / len(d_loader.dataset)
                print("SAVING MODEL")
                self.save(fold)
                # Loggo solo se miglioro 
                if not write_flag:
                    print(conf_matrix.conf_matrix)
                    # Confusion matrix as NumPy array
                    print(type(conf_matrix.conf_matrix))
                    cm_array = conf_matrix.conf_matrix

                    # Plot con seaborn
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_array, annot=True, fmt='d', xticklabels=['0', '0.5', '1', '1.5', '2', '2.5', '3'],
                                yticklabels=['0', '0.5', '1', '1.5', '2', '2.5', '3'], cmap='Blues')
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.title("Confusion Matrix")

                    # Log as image
                    wandb.log({"confusion_matrix_image": wandb.Image(plt)}, step=epoch)
                    plt.close()

        return mae / len(d_loader.dataset), conf_matrix, y_pred_all, images
        

    def thresh_eval(self,epoch):
        self.thresh = 0.5
        print(self.thresh)
     

    def bayesian_dropout_eval(self, dset='test', n_forwards=100, write_flag=False):
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            sigm = nn.Sigmoid()
            sofmx = nn.Softmax(dim=1)
            trues = 0
            tr_trues = 0
            acc = 0
            self.n.eval()
            self.n.dropout.training = True
            conf_tl = []
            logits_tl = []

            tot_samples = 1000.0
            tot_trues = 375

            if dset == 'test':
                eval_loader = self.eval_data_loader

            elif dset == 'validation':
                eval_loader = self.validation_data_loader
                tot_samples = 500.0
                tot_trues = 100
            else:
                print("ERROR")
                return

            for nf in range(n_forwards):
                eval_logit_losses = []
                eval_prob_losses = []
                eval_probs = []
                eval_logits = []
                eval_labels = []
                eval_cat_labels = []

                for i, (x, target, img_name) in enumerate(eval_loader):
                    start_time = time.time()
                    # measure data loading time
                    # print("data time: " + str(time.time() - start_time))

                    # compute output
                    x = x.to('cuda')
                    output = torch.squeeze(self.n(x))
                    print('time for forward: ' + str(time.time() - start_time))
                    start_time = time.time()
                    if self.num_classes == 1:
                        target = target.to('cuda', torch.float)
                        loss = criterion(output, target)
                        eval_logit_losses.append(loss.item())
                        check_output = sigm(output)
                        loss = criterion(check_output, target)
                        eval_prob_losses.append(loss.item())
                        res = (check_output > self.thresh).float()
                    else:
                        target = target.to('cuda', torch.long)
                        # loss = criterion(output, target)
                        # eval_logit_losses.append(loss.item())
                        check_output = sofmx(output)
                        loss = criterion(check_output, target)
                        eval_prob_losses.append(loss.item())
                        eval_probs.append(check_output.to('cpu'))
                        eval_logits.append(output.to('cpu'))
                        eval_labels.append(categorical_to_one_hot(target.to('cpu'), 2))
                        eval_cat_labels.append(target.to('cpu'))
                        check_output, res = torch.max(check_output, 1)

                    print('time for everything else: ' + str(time.time() - start_time))
                if self.num_classes > 1:
                    conf_tl.append(torch.cat(eval_probs, 0))
                    logits_tl.append(torch.cat(eval_logits, 0))
                    acc_t = (torch.cat(eval_labels, 0))
                    acc_cat_t = (torch.cat(eval_cat_labels, 0))

                if self.num_classes > 1 and (nf == 100 or nf == 1000 or nf == 10000):
                    # conf_t = torch.mean(torch.stack(conf_tl, 0), 0)
                    logits_t = torch.mean(torch.stack(logits_tl, 0), 0)

                    # brier = compute_brier(conf_t, acc_t)
                    # cpb, _, samples_per_bin = average_confidence_per_bin(conf_t, 15, False)
                    # apb, _, _ = accuracy_per_bin(conf_t, acc_cat_t, 15, False)
                    # ece, _ = compute_ECE(apb, cpb, samples_per_bin)
                    #
                    # print('logits NLL: ' + str(np.mean(eval_logit_losses)))
                    # print('probs NLL: ' + str(np.mean(eval_prob_losses)))
                    # print('brier: ' + str(brier.item()))
                    # print('confidence per bin: ' + str(cpb))
                    # print('accuracy per bin: ' + str(apb))
                    # print('samples per bin: ' + str(samples_per_bin))
                    # print('ECE: ' + str(ece * 100))

                    if write_flag:
                        with open(files_path + self.nname + '_' + dset + '_' + str(nf) + 'fwds4bayes_preds.csv',
                                  'w') as predsfile, open(
                            files_path + self.nname + '_' + dset + '_' + str(nf) + 'fwds4bayes_gts.csv',
                            'w') as lblsfile:
                            preds_writer = csv.writer(predsfile, delimiter=',')
                            lbls_writer = csv.writer(lblsfile, delimiter=',')
                            for i in range(len(logits_t.squeeze().tolist())):
                                preds_writer.writerow(logits_t.squeeze().tolist()[i])
                                # lbls_writer.writerow(acc_t.squeeze().tolist()[i])

    def validate(self, write_flag=False):
        with torch.no_grad():
            sigm = nn.Sigmoid()
            sofmx = nn.Softmax(dim=1)
            trues = 0
            tr_trues = 0
            acc = 0
            self.n.eval()
            eval_losses = []

            start_time = time.time()
            for i, (x, target, img_name) in enumerate(self.validation_data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to('cuda')
                output = torch.squeeze(self.n(x))
                if self.num_classes == 1:
                    target = target.to('cuda', torch.float)
                    check_output = sigm(output)
                    res = (check_output > self.thresh).float()
                else:
                    target = target.to('cuda', torch.long)
                    check_output = sofmx(output)
                    check_output, res = torch.max(check_output, 1)

                tr_target = target * 2
                tr_target = tr_target - 1
                tr_trues += sum(res == tr_target).item()
                trues += sum(res).item()
                acc += sum(res == target).item()

            pr = tr_trues / (trues + 10e-5)
            rec = tr_trues / 100
            fscore = (2 * pr * rec) / (pr + rec + 10e-5)
            stats_string = 'Test set = Acc: ' + str(acc / 500.0) + ' | F1 Score: ' + str(
                fscore) + ' | Precision: ' + str(
                pr) + ' | Recall: ' + str(rec) + ' | Trues: ' + str(trues) + ' | Correct Trues: ' + str(
                tr_trues) + ' | time: ' + str(time.time() - start_time)
            print(stats_string)

    def explain_eval(self, write_flag=False, target_index=None):
        sigm = nn.Sigmoid()
        sofmx = nn.Softmax(dim=1)
        trues = 0
        tr_trues = 0
        acc = 0
        self.n.eval()
        grad_cam = GradCam(self.n, target_layer_names=["7"], use_cuda=True)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        # target_index = None

        if write_flag:
            self.create_html()

        start_time = time.time()
        for i, (x, target, img_name) in enumerate(self.eval_data_loader_4k):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # compute output
            x = x.to('cuda')
            output = torch.squeeze(self.n(x))
            if self.num_classes == 2:
                target = target.to('cuda', torch.long)
                check_output = sofmx(output)
                check_output, res = torch.max(check_output, 1)
            elif self.num_classes == 7:
                target = target.to('cuda', torch.long)
                check_output = sofmx(output)
                check_output, res = torch.max(check_output, 1)
            else:
                print('unrecognized number of classes')
                exit(1)

            tr_target = target * 2
            tr_target = tr_target - 1
            tr_trues += sum(res == tr_target).item()
            trues += sum(res).item()
            acc += sum(res == target).item()

            # gb_model = GuidedBackprop(self.n)
            for j in range(len(x)):
                in_im = Variable(x[j].unsqueeze(0), requires_grad=True)
                # mask = grad_cam(in_im, 0)
                # show_cam_on_image(nefro.denormalize(x[j]), mask, os.path.basename(img_name[j])[:-4] + self.lbl_name + '_cls0')
                mask = grad_cam(in_im, target_index)
                show_cam_on_image(nefro_4k_and_diapo.denormalize(x[j]), mask,
                                  os.path.basename(img_name[j])[:-4] + str(self.lbl_name) + f'_cls{target_index}')

                # gb = gb_model.generate_gradients(in_im, target_index)
                # save_gradient_images(gb, '/homes/fpollastri/nefro_GradCam/' + os.path.basename(img_name[j])[
                #                                                               :-4] + '_gb.png')
                # cam_gb = np.zeros(gb.shape)
                # if not np.isnan(mask).any():
                #     for c in range(0, gb.shape[0]):
                #         cam_gb[c, :, :] = mask
                #     cam_gb = np.multiply(cam_gb, gb)
                # save_gradient_images(cam_gb, '/homes/fpollastri/nefro_GradCam/' + os.path.basename(img_name[j])[
                #                                                                   :-4] + '_cam_gb.png')
            if write_flag:
                self.write_html(img_name=img_name, target=target, res=res, conf=check_output)
            # # measure elapsed time
            # printProgressBar(i + 1, total + 1,
            #                  length=20,
            #                  prefix=f'Epoch {epoch} ',
            #                  suffix=f', loss: {loss.item():.3f}'
            #                  )
        pr = tr_trues / (trues + 10e-5)
        rec = tr_trues / 375
        fscore = (2 * pr * rec) / (pr + rec + 10e-5)
        stats_string = 'Test set = Acc: ' + str(acc / 1000.0) + ' | F1 Score: ' + str(fscore) + ' | Precision: ' + str(
            pr) + ' | Recall: ' + str(rec) + ' | Trues: ' + str(trues) + ' | Correct Trues: ' + str(
            tr_trues) + ' | time: ' + str(time.time() - start_time)
        print(stats_string)

        if write_flag:
            self.close_html(stats_string)

    def save_outs(self, write_flag=False):
        with torch.no_grad():

            sofmx = nn.Softmax(dim=1)
            trues = 0
            tr_trues = 0
            acc = 0
            n_samples = len(self.eval_data_loader_4k.dataset)
            if self.num_classes == 7:
                outs = np.zeros((n_samples, 2))
            else:
                outs = np.zeros((n_samples, self.num_classes + 1))

            self.n.eval()

            start_time = time.time()
            for i, (x, target, img_name) in enumerate(self.eval_data_loader_4k):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to('cuda')
                output = torch.squeeze(self.n(x))

                target = target.to('cuda', torch.long)
                check_output = sofmx(output)
                _, res = torch.max(check_output, 1)

                tr_target = target * 2
                tr_target = tr_target - 1
                tr_trues += sum(res == tr_target).item()
                trues += sum(res).item()
                acc += sum(res == target).item()
                n_start = i * self.eval_data_loader_4k.batch_size

                if self.num_classes == 7:
                    intensity_values = torch.arange(7).float().cuda()
                    intensity_values /= 2
                    intensity_expect = torch.sum(intensity_values * check_output, 1)
                    t_intensity_expect = torch.round(intensity_expect * 2) / 2
                    temp_out = np.array(t_intensity_expect.cpu())[:, None]
                else:
                    temp_out = np.array(check_output.cpu())

                temp_out = np.hstack((temp_out, np.array(target.cpu()[:, None])))
                outs[n_start:n_start + target.shape[0]] = temp_out

            pr = tr_trues / (trues + 10e-5)
            rec = tr_trues / 375
            fscore = (2 * pr * rec) / (pr + rec + 10e-5)
            stats_string = 'Test set = Acc: ' + str(acc / 1000.0) + ' | F1 Score: ' + str(
                fscore) + ' | Precision: ' + str(
                pr) + ' | Recall: ' + str(rec) + ' | Trues: ' + str(trues) + ' | Correct Trues: ' + str(
                tr_trues) + ' | time: ' + str(time.time() - start_time)
            print(stats_string)
            if write_flag:
                path = '/nas/softechict-nas-2/fpollastri/data/istologia/out_files/'
                np.savetxt(path + str(self.lbl_name) + '_outs.csv', outs, delimiter=',')

    def calibration_write_html(self):
        imgs = np.load(files_path + 'sorted_by_calibrated_probability_' + str(self.lbl_name) + '.npy')
        names, targets = nefro_4k_and_diapo.Nefro.read_csv('test', self.lbl_name)
        gts = []
        preds = []
        with open(files_path + 'densenet_' + str(self.lbl_name) + '_test_gts.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                gts.append(int(row[1][0]))

        with open(files_path + 'densenet_' + str(self.lbl_name) + '_test_preds.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                p0, p1 = float(row[0]), float(row[1])
                preds.append(np.argmax(np.array([p0, p1])))

        ninds = imgs[:, 0].astype(int)
        pnames = np.array(names)
        pnames = pnames[ninds]
        ppreds = np.array(preds)
        ppreds = ppreds[ninds]
        pgts = np.array(gts)
        pgts = pgts[ninds]
        self.create_html(csvfname='_calibration')

        self.write_html_cal(img_name=pnames, cal=imgs[:, 1], uncal=imgs[:, 2], preds=ppreds, gts=pgts,
                            csvfname='_calibration')

        self.close_html('', csvfname='_calibration')

    def paper_figure(self):
        imgs = np.load(files_path + 'sorted_by_calibrated_probability_' + self.lbl_name + '.npy')
        names, targets = nefro_4k_and_diapo.Nefro.read_csv('test', self.lbl_name)
        gts = []
        preds = []
        with open(files_path + 'densenet_' + self.lbl_name + '_test_gts.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                gts.append(int(row[1][0]))

        with open(files_path + 'densenet_' + self.lbl_name + '_test_preds.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                p0, p1 = float(row[0]), float(row[1])
                preds.append(np.argmax(np.array([p0, p1])))

        ninds = imgs[:, 0].astype(int)
        pnames = np.array(names)
        pnames = pnames[ninds]
        ppreds = np.array(preds)
        ppreds = ppreds[ninds]
        pgts = np.array(gts)
        pgts = pgts[ninds]

        pnames = ['7F5ZB850_F00024770.tif', '7F5ZB850_F00011264.tif', '7F5ZB850_F00017607.tif',
                  '7F5ZB850_F00009726.tif', '7F5ZB850_F00022664.tif']

        self.create_html(csvfname='_papefig_yy')

        self.write_html_cal(img_name=pnames, cal=imgs[:, 1], uncal=imgs[:, 2], preds=ppreds, gts=pgts,
                            csvfname='_papefig_yy')

        self.close_html('', csvfname='_papefig_yy')

    def eval_calibration(self, dset='test', write_flag=False):
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            sigm = nn.Sigmoid()
            sofmx = nn.Softmax(dim=1)
            trues = 0
            tr_trues = 0
            acc = 0
            self.n.eval()
            eval_logit_losses = []
            eval_prob_losses = []
            eval_probs = []
            eval_logits = []
            eval_labels = []
            eval_cat_labels = []

            tot_samples = 1000.0
            tot_trues = 375

            if dset == 'test':
                eval_loader = self.eval_data_loader

            elif dset == 'validation':
                eval_loader = self.validation_data_loader
                tot_samples = 500.0
                tot_trues = 100
            else:
                print("ERROR")
                return
            start_time = time.time()
            for i, (x, target, img_name) in enumerate(eval_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to('cuda')
                output = torch.squeeze(self.n(x))
                if self.num_classes == 1:
                    target = target.to('cuda', torch.float)
                    loss = criterion(output, target)
                    eval_logit_losses.append(loss.item())
                    check_output = sigm(output)
                    loss = criterion(check_output, target)
                    eval_prob_losses.append(loss.item())
                    res = (check_output > self.thresh).float()
                else:
                    target = target.to('cuda', torch.long)
                    loss = criterion(output, target)
                    eval_logit_losses.append(loss.item())
                    check_output = sofmx(output)
                    loss = criterion(check_output, target)
                    eval_prob_losses.append(loss.item())
                    eval_probs.append(check_output.to('cpu'))
                    eval_logits.append(output.to('cpu'))
                    eval_labels.append(categorical_to_one_hot(target.to('cpu'), 2))
                    eval_cat_labels.append(target.to('cpu'))
                    check_output, res = torch.max(check_output, 1)

                tr_target = target * 2
                tr_target = tr_target - 1
                tr_trues += sum(res == tr_target).item()
                trues += sum(res).item()
                acc += sum(res == target).item()

            pr = tr_trues / (trues + 10e-5)
            rec = tr_trues / tot_trues
            fscore = (2 * pr * rec) / (pr + rec + 10e-5)
            stats_string = 'Test set = Acc: ' + str(acc / tot_samples) + ' | F1 Score: ' + str(
                fscore) + ' | Precision: ' + str(
                pr) + ' | Recall: ' + str(rec) + ' | Trues: ' + str(trues) + ' | Correct Trues: ' + str(
                tr_trues) + ' | time: ' + str(time.time() - start_time)
            print(stats_string)

            if self.num_classes > 1:
                conf_t = torch.cat(eval_probs, 0)
                logits_t = torch.cat(eval_logits, 0)
                acc_t = torch.cat(eval_labels, 0)
                acc_cat_t = torch.cat(eval_cat_labels, 0)

                brier = compute_brier(conf_t, acc_t)
                cpb, _, samples_per_bin = average_confidence_per_bin(conf_t, 15, False)
                apb, _, _ = accuracy_per_bin(conf_t, acc_cat_t, 15, False)
                ece, _ = compute_ECE(apb, cpb, samples_per_bin)

                print('logits NLL: ' + str(np.mean(eval_logit_losses)))
                print('probs NLL: ' + str(np.mean(eval_prob_losses)))
                print('brier: ' + str(brier.item()))
                print('confidence per bin: ' + str(cpb))
                print('accuracy per bin: ' + str(apb))
                print('samples per bin: ' + str(samples_per_bin))
                print('ECE: ' + str(ece * 100))

                if write_flag:
                    with open(files_path + self.nname + '_' + dset + '_preds.csv', 'w') as predsfile, open(
                            files_path + self.nname + '_' + dset + '_gts.csv',
                            'w') as lblsfile:
                        preds_writer = csv.writer(predsfile, delimiter=',')
                        lbls_writer = csv.writer(lblsfile, delimiter=',')
                        for i in range(len(conf_t.squeeze().tolist())):
                            preds_writer.writerow(logits_t.squeeze().tolist()[i])
                            lbls_writer.writerow(acc_t.squeeze().tolist()[i])

    def create_html(self, csvfname=''):
        ffname = files_path + self.net + "_" + str(self.lbl_name) + csvfname + ".html"
        with open(ffname, 'w+') as f:
            f.write(
                '<!DOCTYPE html> <html> <head> <title>' + os.path.basename(
                    ffname) + '</title> </head> <body> <h1>' + self.net + ' on ' + str(
                    self.lbl_name) + '</h1>' + '<table>'
                # + '<col width="1000"><col width="1000"><col width="1000"><col width="1000"><col width="1000">' +
                # '<col width="1000">'
            )

    def write_html(self, img_name, target, res, conf, csvfname=''):
        ffname = files_path + self.net + "_" + str(self.lbl_name) + csvfname + ".html"
        for i in range(len(img_name)):
            pngname = 'png/' + os.path.basename(img_name[i])[:-4] + '.png'
            cam0name = 'GradCam/' + os.path.basename(img_name[i])[:-4] + str(self.lbl_name) + '_cls0_cam.png'
            cam1name = 'GradCam/' + os.path.basename(img_name[i])[:-4] + str(self.lbl_name) + '_cls1_cam.png'
            correct = target[i] == res[i]
            lbl = get_label(target[i], correct)
            pred = get_label(res[i], correct)
            confidence = get_conf(conf[i].item(), correct)
            with open(ffname, 'a') as f:
                f.write(
                    '<tr>' +
                    ' <td> <font size=\"7\">' + os.path.basename(img_name[i])[:-4] + '</font> </td>' +
                    ' <td> <img src=\"' + pngname + '\" width=\"512px\" height=\"512px\"> </td>' +
                    ' <td> <img src=\"' + cam1name + '\" width=\"512px\" height=\"512px\"> </td>' +
                    ' <td> <img src=\"' + cam0name + '\" width=\"512px\" height=\"512px\"> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> ' + str(self.lbl_name) + '</font></td> </tr>' + lbl +
                    ' </table> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> Predizione </font> </td> </tr>' + pred + '</table> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> Confidence </font> </td> </tr>' + confidence + '</table> </td>' +
                    '</tr>')

    def write_html_cal(self, img_name, cal, uncal, preds, gts, csvfname=''):
        ffname = files_path + self.net + "_" + str(self.lbl_name) + csvfname + ".html"
        for i in range(len(img_name)):
            pngname = 'nefro_png/' + img_name[i][:-4] + '.png'
            correct = (preds[i]) == (gts[i])
            lbl = get_label(gts[i], correct)
            pred = get_label(preds[i], correct)
            with open(ffname, 'a') as f:
                f.write(
                    '<tr>' +
                    ' <td> <font size=\"7\">' + img_name[i][:-4] + '</font> </td>' +
                    ' <td> <img src=\"' + pngname + '\" width=\"512px\" height=\"512px\"> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> Calibrated </font> </td> </tr>' + '<tr>' +
                    '<td> <font size=\"7\">' + str(cal[i])[:5] + '</font> </td>' +
                    '</tr>' + '</table> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> Uncalibrated </font> </td> </tr>' + '<tr>' +
                    '<td> <font size=\"7\">' + str(uncal[i])[:5] + '</font> </td>' +
                    '</tr>' + '</table> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> ' + 'GroundTruth' + '</font></td> </tr>' + '<tr>' +
                    lbl +
                    '</tr>' + ' </table> </td>' +
                    '<td> <table> <tr> <td> <font size=\"7\"> Prediction </font> </td> </tr>' + '<tr>' +
                    pred +
                    '</tr>' + '</table> </td>' +
                    '</tr>')

    def close_html(self, stats, csvfname=''):
        ffname = files_path + self.net + "_" + str(self.lbl_name) + csvfname + ".html"
        with open(ffname, 'a') as f:
            f.write('</table> <h1> ' + stats + ' </h1> </body> </html>')

    def write_csv(self, img_name, target, res, conf):
        for i in range(len(img_name)):
            with open(files_path + 'ROC.csv', 'a') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                csv_writer.writerow([img_name[i], str(int(target[i])), str(int(res[i])), conf[i].item()])

    def find_stats(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        b = 0
        for data, _, _ in self.data_loader:
            b += 1
            print(b)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntraining")
        print("mean: " + str(mean) + " | std: " + str(std))

    def incrementForFilename(self, prefix_name):
        cwd = self.models_dir
        i = 0
        # faccio il loop finchè non trovo un i che non esiste
        while FilePath(cwd, f"{prefix_name}{i}_net.pth").is_file():
            i += 1
        return i 

    def get_probabilities(self, dl):
        # Se il dataset è un Subset, allora accediamo al dataset vero
        if isinstance(dl.dataset, torch.utils.data.Subset):
            dataset = dl.dataset.dataset
            indices = dl.dataset.indices
            # Ottieni solo le label dagli indici
            lbls = [dataset.lbls[i] for i in indices]
        else:
            lbls = dl.dataset.lbls

        counter = sum(lbls)
        total = len(lbls)
        print(f"Classe 1 count: {counter} / {total} ({counter/total:.4f})")
        
        c1_w = counter / total
        epsilon = 1e-6
        c1_w = min(max(c1_w, epsilon), 1 - epsilon)

        c0_w = 1.0 - c1_w
        c1_w = 1.0 / c1_w
        c0_w = 1.0 / c0_w
        return c0_w, c1_w

    
    # Qui bisogna salvare i pesi con un identificativo di qualche tipo
    # def save(self, fold):
    #     # Questo serve per identificare se faccio esperimenti old o new 
    #     if self.old_or_new_folder == 'Files_old_Pollo/':
    #         nname = str(self.net) + '_' + str(self.lbl_name) + '_Old'
    #     else:
    #         nname = str(self.net) + '_' + str(self.lbl_name) + '_New'

    #     if fold != None:
    #         nname = nname + '_fold_' + str(fold) + '_'

    #     if self.dropout:
    #         nname = 'dropout_' + nname
    #         # Questo serve per controllare se ci sono già pesi per quella classe e se ci sono mettere un numero incrementale per non sovrascriverli
    #         # Questo per la prima epoca chiaramente, poi un volta creati i pesi nuovi si sovrascrivono
    #     if not hasattr(self, 'base_index'):
    #         i = self.incrementForFilename(nname)
    #         self.base_index = i
            
    #     try:
    #         save_dir_n = os.path.join(self.models_dir, nname + str(self.base_index) + '_net.pth')
    #         torch.save(self.n.state_dict(), save_dir_n)
    #         save_dir_optimizer = os.path.join(self.models_dir, nname + str(self.base_index) + '_opt.pth')
    #         torch.save(self.optimizer.state_dict(), save_dir_optimizer)
    #         print(f"Model weights successfully saved in {save_dir_n}")
    #         return True
    #     except Exception as e:
    #         print(f"Error during Saving: {e}")
    #         return False

    def save(self, fold=None):

        """
        Salva lo stato del modello e dell'optimizer.
        Se fold è specificato, salva in una sottocartella dedicata al fold.
        Altrimenti salva nella cartella principale con un indice.
        """

        # Costruisci nome base
        if self.old_or_new_folder == 'Files_old_Pollo/':
            nname = f"{self.net}_{self.lbl_name}_Old"
        else:
            nname = f"{self.net}_{self.lbl_name}_New"

        if self.dropout:
            nname = f"dropout_{nname}"

        # CASO 1: Cross-validation (fold specificato)
        if fold is not None:
            print(f'Il valore di fold è {fold}')
            # Crea sottocartella per il fold
            fold_subdir = os.path.join(self.models_dir, f"fold_{fold}")
            os.makedirs(fold_subdir, exist_ok=True)

            try:
                model_path = os.path.join(fold_subdir, f"{nname}_{fold}_net.pth")
                opt_path = os.path.join(fold_subdir, f"{nname}_{fold}_opt.pth")

                torch.save(self.n.state_dict(), model_path)
                torch.save(self.optimizer.state_dict(), opt_path)

                print(f"Model weights saved for fold {fold} in {fold_subdir}")
                return True
            except Exception as e:
                print(f"Error during saving fold {fold}: {e}")
                return False

        # CASO 2: Salvataggio standard (no fold)
        else:
            print(f'Il valore di fold è {fold}, perchè non sto facendo cross-validation')
            if not hasattr(self, 'base_index'):
                i = self.incrementForFilename(nname)
                self.base_index = i

            try:
                model_path = os.path.join(self.models_dir, f"{nname}_{self.base_index}_net.pth")
                opt_path = os.path.join(self.models_dir, f"{nname}_{self.base_index}_opt.pth")

                torch.save(self.n.state_dict(), model_path)
                torch.save(self.optimizer.state_dict(), opt_path)

                print(f"Model weights saved in {model_path}")
                return True
            except Exception as e:
                print(f"Error during standard saving: {e}")
                return False

        
    def load(self, data):
        nname = self.net + '_' + str(self.lbl_name) + data
        if self.dropout:
            nname = 'dropout_' + nname
        w_path = os.path.join(self.models_dir, nname + '_net.pth')
        self.n.load_state_dict(torch.load(w_path))
        #self.optimizer.load_state_dict(torch.load(os.path.join(self.models_dir, nname + '_opt.pth')))
        print(f"model weights successfully loaded: {self.n.load_state_dict(torch.load(w_path))} ")
        print('Weights path : ', w_path)
        return w_path

    def load_old_ckpt(self, ckpt_name='_old'):
        self.n.load_state_dict(torch.load(os.path.join(self.models_dir, self.lbl_name + '_net' + ckpt_name + '.pth')))
        # self.optimizer.load_state_dict(torch.load(os.path.join(self.models_dir, self.lbl_name + '_opt' + ckpt_name + '.pth')))
        print("model old weights successfully loaded")

    def see_imgs(self):
        cntr = 0
        for data in self.eval_data_loader:
            cntr += 1
            save_image(data[0].float(),
                       '/homes/fpollastri/aug_images/' + os.path.basename(data[2][0])[:-4] + '.png',
                       nrow=1, pad_value=0)
            print("img saved")

    def write_testset(self):
        eval_dataset = nefro_4k_and_diapo.Nefro(load=False, split_name='test', label_name=self.lbl_name,
                                                size=(opt.size, opt.size),
                                                transform=transforms.Compose([
                                                    transforms.Resize((self.size, self.size)),
                                                    nefro_4k_and_diapo.NefroTiffToTensor(),
                                                    # transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851)),
                                                ])
                                                )
        eval_data_loader = DataLoader(eval_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=self.n_workers,
                                      drop_last=False,
                                      pin_memory=True)

        cntr = 0
        for data in eval_data_loader:
            cntr += 1
            data[0][0, 0, :, :] = 0
            data[0][0, 2, :, :] = 0
            save_image(data[0].float(),
                       '/nas/softechict-nas-1/fpollastri/nefro_png//' + os.path.basename(data[2][0])[:-4] + '.png',
                       nrow=1, pad_value=0)
            print(os.path.basename(data[2][0])[:-4] + "saved")
        print(cntr)

    def write_images(self, imgs):
        eval_dataset = nefro_4k_and_diapo.Nefro(load=False, split_name='validation', label_name=self.lbl_name,
                                                size=(opt.size, opt.size),
                                                transform=transforms.Compose([
                                                    transforms.Resize((self.size, self.size)),
                                                    nefro_4k_and_diapo.NefroTiffToTensor(),
                                                    # transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851)),
                                                ])
                                                )
        eval_data_loader = DataLoader(eval_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=self.n_workers,
                                      drop_last=False,
                                      pin_memory=True)

        cntr = 0
        for data in eval_data_loader:
            if os.path.basename(data[2][0])[:-4] == imgs:
                cntr += 1
                data[0][0, 0, :, :] = 0
                data[0][0, 2, :, :] = 0
                save_image(data[0].float(),
                           '/nas/softechict-nas-1/fpollastri/nefro_png//' + os.path.basename(data[2][0])[:-4] + '.png',
                           nrow=1, pad_value=0)
                print(os.path.basename(data[2][0])[:-4] + "saved")
        print(cntr)

# Funzione per pesare la Loss function 
def get_weights(target):
    # 0.9 for True, 0.2 for Falses
    #weights = target * 0.7
    weights = target * 0.2
    weights += 0.
    return weights

def split_dataset(label, w4k=False, wdiapo=False, n_trues_test=375, n_test=1000, n_trues_val=100, n_val=200,
                  custom_name=""):
    # dataset = nefro.Nefro(load=False)
    if w4k:
        if wdiapo:
            csv_file_name = nefro_4k_and_diapo.Nefro.splitsdic.get('images')
        else:
            csv_file_name = nefro_4k_and_diapo.Nefro.splitsdic.get('4k')
    elif wdiapo:
        csv_file_name = nefro_4k_and_diapo.Nefro.splitsdic.get('diapo')
    else:
        raise ValueError("no dataset to split (w4k and wdiapo are both set to False)")

    # dataset = nefro_4k_and_diapo.Nefro(split='everything', label_name=label)
    # lbl_idx = dataset.flagsdic.get(label)
    lbl_idxs = [nefro_4k_and_diapo.Nefro.flagsdic.get(l) for l in label[0]]
    d = {}
    trues = 0
    falses = 0
    training = {}
    test = {}
    val = {}
    with open(csv_file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            t_flag = False
            for lbl_idx in lbl_idxs:
                if row[lbl_idx] == 'True' or row[lbl_idx] == 'TRUE':
                    t_flag = True
            d[row[0]] = str(t_flag)
    print(len(d))
    while len(test) < n_test:
        n = random.choice(list(d.keys()))
        if (d.get(n) == 'True' or d.get(n) == 'TRUE') and trues < n_trues_test and n not in test:
            test[n] = d.get(n)
            trues += 1
        elif (d.get(n) == 'False' or d.get(n) == 'FALSE') and falses < 1000 - n_trues_test and n not in test:
            test[n] = d.get(n)
            falses += 1

    s = set(test.keys())
    print('test:' + str(len(s)))
    trues = 0
    falses = 0

    while len(val) < n_val:
        n = random.choice(list(d.keys()))
        if (d.get(n) == 'True' or d.get(n) == 'TRUE') and trues < n_trues_val and n not in test and n not in val:
            val[n] = d.get(n)
            trues += 1
        elif (d.get(n) == 'False' or d.get(
                n) == 'FALSE') and falses < 500 - n_trues_val and n not in test and n not in val:
            val[n] = d.get(n)
            falses += 1

    s = set(val.keys())
    print('val:' + str(len(s)))

    for el in d:
        if el not in test and el not in val:
            training[el] = d[el]
    s = set(training.keys())
    print('training:' + str(len(s)))

    with open(csv_file_name) as csvfile, open(
            nefro_4k_and_diapo.Nefro.get_split_name(label, w4k, wdiapo, 'test', custom_name), 'w',
            newline='') as testfile, open(
        nefro_4k_and_diapo.Nefro.get_split_name(label, w4k, wdiapo, 'training', custom_name), 'w',
        newline='') as trainfile, open(
        nefro_4k_and_diapo.Nefro.get_split_name(label, w4k, wdiapo, 'validation', custom_name), 'w',
        newline='') as valfile:
        # open(files_path + split_name + label + '_test.csv', 'w', newline='') as testfile,
        readCSV = csv.reader(csvfile, delimiter=',')
        test_writer = csv.writer(testfile, delimiter=',')
        train_writer = csv.writer(trainfile, delimiter=',')
        val_writer = csv.writer(valfile, delimiter=',')
        for row in readCSV:
            if row[0] not in d:
                continue
            else:
                del d[row[0]]
            if row[0] in test:
                test_writer.writerow(row)
            elif row[0] in val:
                val_writer.writerow(row)
            elif row[0] in training:
                train_writer.writerow(row)
            else:
                print(row)


def merge_datasets(label):
    splits = ['training', 'validation', 'test']

    for split in splits:
        names_l = []
        names_l.append(nefro_4k_and_diapo.Nefro.get_split_name(label=label, w4k=True, wdiapo=False, split=split))
        names_l.append(nefro_4k_and_diapo.Nefro.get_split_name(label=label, w4k=False, wdiapo=True, split=split))
        new_name = nefro_4k_and_diapo.Nefro.get_split_name(label=label, w4k=True, wdiapo=True, split=split)

        with open(new_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            for name in names_l:
                with open(name) as readfile:
                    readCSV = csv.reader(readfile, delimiter=',')
                    for row in readCSV:
                        csv_writer.writerow(row)


def split_dataset_morelabels(labels, name='multilabels'):
    dataset = nefro_4k_and_diapo.Nefro(load=False)
    lbl_idxs = []
    for label in labels:
        lbl_idxs.append(dataset.flagsdic.get(label))
    d = {}
    trues = 0
    falses = 0
    training = {}
    val = {}
    test = {}
    t_count = 0
    with open(files_path + 'whole_dataset_labels.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            d[row[0]] = 'False'
            for lbl_idx in lbl_idxs:
                if row[lbl_idx] == 'True':
                    d[row[0]] = row[lbl_idx]
                    t_count += 1
                    break
    print(len(d))
    print('of wich ' + str(t_count) + ' are true')

    while len(test) < 1000:
        n = random.choice(list(d.keys()))
        if d.get(n) == 'True' and trues < 450 and n not in test:
            test[n] = d.get(n)
            trues += 1
        elif d.get(n) == 'False' and falses < 550 and n not in test:
            test[n] = d.get(n)
            falses += 1

    s = set(test.keys())
    print('test:' + str(len(s)))
    trues = 0
    falses = 0

    while len(val) < 500:
        n = random.choice(list(d.keys()))
        if d.get(n) == 'True' and trues < 150 and n not in test and n not in val:
            val[n] = d.get(n)
            trues += 1
        elif d.get(n) == 'False' and falses < 350 and n not in test and n not in val:
            val[n] = d.get(n)
            falses += 1

    s = set(val.keys())
    print('val:' + str(len(s)))

    for el in d:
        if el not in test and el not in val:
            training[el] = d[el]
    s = set(training.keys())
    print('training:' + str(len(s)))

    with open(files_path + 'whole_dataset_labels.csv') as csvfile, \
            open(files_path + name + '_test.csv', 'w', newline='') as testfile, \
            open(files_path + name + '_training.csv', 'w', newline='') as trainfile, \
            open(files_path + name + '_validation.csv', 'w', newline='') as valfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        test_writer = csv.writer(testfile, delimiter=',')
        train_writer = csv.writer(trainfile, delimiter=',')
        val_writer = csv.writer(valfile, delimiter=',')
        for row in readCSV:
            if row[0] not in d:
                continue
            else:
                del d[row[0]]
            if row[0] in test:
                test_writer.writerow((row[0], test.get(row[0])))
            elif row[0] in val:
                val_writer.writerow((row[0], val.get(row[0])))
            elif row[0] in training:
                train_writer.writerow((row[0], training.get(row[0])))
            else:
                print(row)


def write_on_file(epoch, training_cost, validation_acc, t_validation_acc, tm, filename):
    ffname = files_path + filename
    with open(ffname, 'a+') as f:
        f.write("E:" + str(epoch) + " | Time: " + str(tm) +
                " | Training_acc: " + str(1 - training_cost) +
                " | validation_acc: " + str(1 - validation_acc) +
                " | t_validation_acc: " + str(t_validation_acc) + "\n")


def get_label(t, corr):
    if t:
        return '<td><font size="7" color=\"' + get_color(corr) + '\"><b> Yes </b></font></td>'
    return '<td><font size="7" color=\"' + get_color(corr) + '\"><b> No </b></font></td>'


def get_conf(t, corr):
    if t > 0.5:
        return '<tr> <td><font size="7" color=\"' + get_color(corr) + '\"><b>' + f"{t:.4f}" + '</b></font></td> </tr>'
    return ' <tr> <td><font size="7" color=\"' + get_color(corr) + '\"><b>' + \
           f"{(1.0 - t):.4f}" + ' </b></font></td> </tr>'


def get_color(corr):
    if corr:
        return "00FF00"
    return "FF0000"


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.moveaxis(np.float32(img.cpu()), 0, -1)
    cam = cam / np.max(cam)
    cv2.imwrite('/homes/fpollastri/nefro_GradCam/' + name + '_cam.png', np.uint8(255 * cam))


def plot(img):
    return
    plt.figure()
    # plt.imshow(nefro_4k_and_diapo.denormalize(img))
    plt.imshow(img)
    plt.show(block=False)


if __name__ == '__main__':

    os.environ["OMP_NUM_THREADS"] = "1"
    def setup_seeds(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        ia.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[SEED] All random seeds set to {seed} for deterministic behavior")

    files_path = '/nas/softechict-nas-1/fpollastri/data/istologia/files//'
    # Cartella in cui salvo i pesi del training
    models_dir = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain"
    # Percorso pesi per fine-tuning
    weights_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/resnet18_[['MESANGIALE']]_Old10_net.pth"

    parser = argparse.ArgumentParser(description='Train ResNet on Glomeruli Labels')
    parser.add_argument('--label', type=str, default='GLOBAL_SEGMENTAL', help='Label group name (e.g., MESANGIALE, LIN_PSEUDOLIN, etc.)')
    parser.add_argument('--old_or_new_dataset_folder', type= str, default = 'Files/', help='use old Pollastri dataset or new Magistroni, Files_old_Pollo/ or Files/')
    parser.add_argument('--train_or_test', type=str, default='Test_on_folds', help='Train or test on your data')
    parser.add_argument('--weights', type=str, default='', help='Name of weights to be loaded')
    parser.add_argument('--conf_matrix_label', type=str, nargs='+', default=['0', '0.5', '1', '1.5', '2', '2.5', '3'],help='Etichette da mostrare nella matrice di confusione')
    parser.add_argument('--network', default='resnet18')
    parser.add_argument('--project_name', default='Train_ResNet_18')
    parser.add_argument('--dropout', action='store_true', help='DropOut')
    parser.add_argument('--wandb_flag', type=bool, default=False, help='wand init')
    parser.add_argument('--val_loss', type=str, default='False', help='log_validation_loss')
    parser.add_argument('--sampler', type=str, default='False', help='use sampler or not')
    parser.add_argument('--classes', type=int, default=2, help='number of classes to train')
    parser.add_argument('--wloss', type=str, default='True', help='weighted or not loss')
    parser.add_argument('--loadEpoch', type=int, default=0, help='load pretrained models')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size during the training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--scheduler', type=str, default='OneCycle', help='scheduler')
    parser.add_argument('--thresh', type=float, default=0.5, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--size', type=int, default=512, help='size of images')
    # parser.add_argument('--w4k', action='store_true', help='is training on 4k dataset')
    parser.add_argument('--w4k', type=bool, default=True, help='is training on 4k dataset')
    parser.add_argument('--load_for_fine_tuning', type=str, default='False', help='load weights for fine tuning')
    parser.add_argument('--wdiapo', action='store_true', help='is training on diapo dataset')
    parser.add_argument('--n_forwards', type=int, default=1, help='number of different forwards to compute')
    parser.add_argument('--savemodel', type=int, default=5, help='number of epochs between saving models')
    parser.add_argument('--SRV', type=bool, default=True, help='is training on remote server')
    parser.add_argument('--from_scratch', action='store_true', help='not finetuning')
    parser.add_argument('--augm_config', type=int, default=0, help='configuration code for augmentation techniques choice')
    """
    La randomicità locale di ogni immagine è diversa perché richiamo il reseed() (nella call delle augmentation) con un seed diverso ogni volta (che però deriva da un random controllato).
    La riproducibilità globale rimane perché il generatore di numeri casuali globale è fissato con il seed nel main."""

    parser.add_argument('--seed', type=int, default=42, help = 'seed for transformation')
    opt = parser.parse_args()

    # LOGICA LABEL SELECTION
    if opt.label == 'MESANGIALE':
        labels_to_use = [['MESANGIALE']]
    elif opt.label == 'LIN_PSEUDOLIN':
        labels_to_use = [['LIN', 'PSEUDOLIN']]
    elif opt.label == 'INTENS':
        labels_to_use = [['INTENS']]
    elif opt.label == 'GEN_SEGM_FOC_SEGM':
        labels_to_use = [['GEN_SEGM', 'FOC_SEGM']]
    elif opt.label == 'GEN_DIFF_FOC_GLOB':
        labels_to_use = [['GEN_DIFF', 'FOC_GLOB']]
    elif opt.label == 'GRAN_FINE':
        labels_to_use = [['GRAN_FINE']]
    elif opt.label == 'GRAN_GROSS':
        labels_to_use = [['GRAN_GROSS']]
    elif opt.label == 'PAR_REGOL_CONT':
        labels_to_use = [['PAR_REGOL_CONT']]
    elif opt.label == 'PAR_REGOL_DISCONT':
        labels_to_use = [['PAR_REGOL_DISCONT']]
    elif opt.label == 'PAR_IRREG':
        labels_to_use = [['PAR_IRREG']]
    elif opt.label == 'GLOBAL_SEGMENTAL':
        labels_to_use = [['GLOB', 'SEGM']]
    elif opt.label == 'GLOBAL':
        labels_to_use = [['GLOB']]
    elif opt.label == 'SEGMENTAL':
        labels_to_use = [['SEGM']]
    else:
        raise ValueError(f"Label group '{opt.label}' non riconosciuto.")
    print(opt)

    setup_seeds(opt.seed)

    if opt.SRV:

        n = NefroNet(net=opt.network, project_name=opt.project_name, wloss = opt.wloss, old_or_new_folder = opt.old_or_new_dataset_folder, 
                     dropout=opt.dropout, wandb_flag=opt.wandb_flag, sampler=opt.sampler, num_classes=opt.classes, num_epochs=opt.epochs,
                     size=opt.size, val_loss=opt.val_loss, batch_size=opt.batch_size, thresh=opt.thresh, pretrained=(not opt.from_scratch),
                     l_r=opt.learning_rate, scheduler=opt.scheduler, n_workers=opt.workers, lbl_name=labels_to_use, conf_matrix_lbl=opt.conf_matrix_label, 
                     w4k=opt.w4k, wdiapo=opt.wdiapo, load_for_fine_tuning = opt.load_for_fine_tuning, weights_path = weights_path, models_dir = models_dir,
                     write_flag=False)
        
        # CLASSIC TRAINING
        if opt.train_or_test == 'Train':
            if opt.label == 'INTENS' :
                # Per training sui fold
                n.train_test_on_folds()
                # Per training su uno split specifico 
                #n.train_intensity()
            else:
                # Per training sui fold
                n.train_test_on_folds()
                # Per training su uno split specifico, per ora bisogna ancora generare lo split con lo script label_generator_from_excel e poi copiare lo split dentro ai csv della classe corretta
                #n.train()
            
        elif opt.train_or_test == 'Test':
            result_path = f'/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Results/result_Seed16_{opt.label}.json'
            if opt.label != 'INTENS':
                data = opt.weights
                w_path = n.load(data)
                accuracy, pr, rec, fscore, cm = n.eval(n.eval_data_loader_4k, True)
                cm_pretty = f"""[[TN={cm[0,0]} FP={cm[0,1]}]
                                [FN={cm[1,0]} TP={cm[1,1]}]]"""
                res_dict = {
                    'Commento' : 'Esperimento su nuovi dati, Sampler, con WLoss, con Max su output, Seed 42 per random, lr 0.1, canali RGB, seed 16 per split, salvo pesi con pazienza 15 epoche',
                    'Esperimento': vars(opt),
                    'Accuracy': float(accuracy),
                    'Precision': float(pr),
                    'Recall': float(rec),
                    'Fscore': float(fscore),
                    "Conf_matrix": cm_pretty,
                    "Weights" : w_path
                }
                if os.path.exists(result_path):
                    with open(result_path, 'r') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_results = data
                            else:
                                all_results = [data]
                        except json.JSONDecodeError:
                            all_results = []
                else:
                    all_results = []
                all_results.append(res_dict)

                with open(result_path, 'w') as f:
                    json.dump(all_results, f, indent=4)

                print("Risultato aggiunto al file JSON.")
            else: 
                    print('Qua ci vuole la eval_intensity')
                    # n.load()
                    # n.eval_intensity(n.validation_data_loader, 0)
                
        elif opt.train_or_test == 'Test_on_folds':
            """Questa funzione serve per fare una evaluation seprata sui fold su cui ho trainato con n.train_test_on_folds(), 
               i risultati vengono scritti dentro a result_path dove ho sia il nome del glomerulo sia l'output della rete per quella immagine.
            """
            result_path = f"/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Results_with_images/{opt.label}"
            folds_weights_pth = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/Folds'
            n.separeted_evaluation_on_folds(folds_weights_pth, result_path)

        
   














































        ##########################################
        # n.explain_eval(True)
        # #n.eval_calibration(dset='validation', write_flag=True)
        # n.eval_calibration(dset='test', write_flag=True)
        # n.find_stats()
        # n.see_imgs()
        # n.write_testset()
    # else:
    #     n = NefroNet(net=opt.network, dropout=opt.dropout, num_classes=opt.classes, num_epochs=opt.epochs,
    #                  size=opt.size, batch_size=opt.batch_size, thresh=opt.thresh, pretrained=(not opt.from_scratch),
    #                  l_r=opt.learning_rate, n_workers=opt.workers, lbl_name=opt.label, w4k=opt.w4k, wdiapo=opt.wdiapo,
    #                  write_flag=False)
    #     n.load()
    #     n.explain_eval(True)
        # n.eval(n.validation_data_loader, epoch=1, write_flag=False)
        # split_dataset('Par-regol-cont', 200, 50)

        # n.load_old_ckpt('_few_epochs')
        # n.calibration_write_html()
        # n.paper_figure()
        # n.write_testset()
        # n.write_images('7F5ZB850_F00002948')
        # n.write_images(['7F5ZB850_F00012734', '7F5ZB850_F00002948'])
        # n.bayesian_dropout_eval(dset='validation', n_forwards=opt.n_forwards, write_flag=False)
        # n.bayesian_dropout_eval(dset='test', n_forwards=opt.n_forwards, write_flag=False)
        # n.eval()
        # n.eval_calibration(dset='validation', write_flag=True)
        # n.eval_calibration(dset='test', write_flag=True)


        # Percorso alle immagini 
        # /nas/softechict-nas-1/fpollastri/data/istologia/images

        # MESANGIALE CON PESI DI POLLASTRI IN EVALUATION (num_classes = 2)
        #         Questa è la confusion matrix :  [620   80]
                                               #  [ 97  203]
        # Acc: 0.8230 | F1 Score: 0.6964 | Precision: 0.7173 | Recall: 0.6767 | Ground Truth Trues: 300 | Time: 9.75s

        # MESANGIALE CON TRAINING SUI VECCHI DATI (num_classes = 2) ed EVAL SUI DATI VECCHI (pesi Old2) usando MAX in output
        # Train_r18_Old_Data_MaxOut_CrossWLoss_Num_classes_2_[['MESANGIALE']]_2025-07-08
                    # Questa è la confusion matrix :    [498 202]
                                                     #  [ 97 203]
        # Acc: 0.7010 | F1 Score: 0.5759 | Precision: 0.5012 | Recall: 0.6767 | Ground Truth Trues: 300 | Time: 14.16s

        # MESANGIALE CON TRAINING SUI VECCHI DATI (num_classes = 2) ed EVAL SUI DATI VECCHI (pesi Old0) thresh 0.5
        #         Questa è la confusion matrix :  [467 233]
                                            #     [109 191]
        # Acc: 0.6580 | F1 Score: 0.5276 | Precision: 0.4505 | Recall: 0.6367 | Ground Truth Trues: 300 | Time: 9.88s

        # MESANGIALE CON NUOVO TRAINING ED EVAL SOLO SU DATI NUOVI (num_classes = 1) no Wloss; pesi New1_net.pth
        #         Questa è la confusion matrix :  [  9 106]
                                               #  [  7 188]
        # Acc: 0.6355 | F1 Score: 0.7689 | Precision: 0.6395 | Recall: 0.9641 | Ground Truth Trues: 195.0 | Time: 4.69s

        # MESANGIALE CON NUOVO TRANING SU DATI VECCHI (num_classes = 1) e EVAL SU DATI VECCHI con 80 epoche
        #         Questa è la confusion matrix :  [607 249]
                                               #  [ 93  51]
        # Acc: 0.6580 | F1 Score: 0.2297 | Precision: 0.3542 | Recall: 0.1700 | Ground Truth Trues: 300.0 | Time: 10.01s

        """
        Le tre sotto-categorie che vuoi unire rappresentano diverse morfologie della parete capillare glomerulare, con differenze nel grado di regolarità o continuità, 
        ma tutte sono localizzate nella zona parietale del glomerulo (cioè lungo la parete del capillare o nella regione subendoteliale). 
        Quindi, dal punto di vista anatomico-funzionale o di classificazione più semplificata, è giustificato inglobarle sotto un'unica etichetta "Parietale".
        """

   
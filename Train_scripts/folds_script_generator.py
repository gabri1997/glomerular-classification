from collections import defaultdict
import numpy as np
import random
import os 
import os
import csv
import cv2
import time
import json
from matplotlib.path import Path
from torch.utils.data import Subset
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
import matplotlib.pyplot as plt
import numpy as np
import datetime
import ast
from torch.utils.data import WeightedRandomSampler
import nefro_dataset as nefro_4k_and_diapo


def plot(img):
    return
    plt.figure()
    # plt.imshow(nefro_4k_and_diapo.denormalize(img))
    plt.imshow(img)
    plt.show(block=False)

def extract_base_id(wsi_name):
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


def group_indices_by_base_id(dataset):
    base_id_to_indices = defaultdict(list)

    for idx, img_path in enumerate(dataset.names):
        wsi_name = os.path.basename(os.path.dirname(img_path))  # cartella contenente l’immagine
        base_id = extract_base_id(wsi_name)
        base_id_to_indices[base_id].append(idx)

    return base_id_to_indices

def kfold_train_val_test_split(dataset, k=4, seed=42):
    random.seed(seed)
    base_id_to_indices = group_indices_by_base_id(dataset)

    base_ids = list(base_id_to_indices.keys())
    random.shuffle(base_ids)

    fold_size = len(base_ids) // k
    folds = [base_ids[i * fold_size: (i + 1) * fold_size] for i in range(k)]
    
    results = []
    
    for i in range(k):
        test_ids = folds[i]
        val_ids = folds[(i + 1) % k]  
        train_ids = [bid for j, f in enumerate(folds) if j not in (i, (i + 1) % k) for bid in f]
        
        test_indices = [idx for bid in test_ids for idx in base_id_to_indices[bid]]
        val_indices = [idx for bid in val_ids for idx in base_id_to_indices[bid]]
        train_indices = [idx for bid in train_ids for idx in base_id_to_indices[bid]]

        results.append((
            np.array(train_indices),
            np.array(val_indices),
            np.array(test_indices)
        ))

    return results


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
    
def save_fold_csvs(dataset, folds, output_dir='folds_csv'):
    os.makedirs(output_dir, exist_ok=True)

    for i, (train_idx, val_idx) in enumerate(folds):
        fold_name = f"fold{i+1}"

        def write_csv(indices, filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'label'])
                for idx in indices:
                    img_path = dataset.names[idx]
                    label = dataset.lbls[idx]
                    # adattalo se hai struttura diversa
                    writer.writerow([img_path, label])

        write_csv(train_idx, os.path.join(output_dir, f"{fold_name}_train.csv"))
        write_csv(val_idx, os.path.join(output_dir, f"{fold_name}_val.csv"))

        print(f"Salvati {fold_name}_train.csv e {fold_name}_val.csv")

if __name__ == '__main__':
    

    imgaug_transforms = ImgAugTransform(config_code=0, size=512, SRV=True)
    dataset_for_folds = nefro_4k_and_diapo.Nefro(
                                split='train_val_fused_for_split',
                                old_or_new_folder = 'Files/',
                                label_name=[['MESANGIALE']],
                                w4k=True,
                                wdiapo=False,
                                size=(512, 512),
                                transform=transforms.Compose([
                                    imgaug_transforms,  # questo deve avere __call__ definito
                                    # Questo non ci vuole perchè viene già fatto quando chiamo get_images
                                    #nefro.NefroTiffToTensor(),
                                    #transforms.Normalize((0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))
                                ])
        )
    folds = kfold_train_val_test_split(dataset_for_folds, k=4)

    for f in folds:
        print(f'Fold')
        print(f'Train : {len(f[0])}')
        print(f'Val : {len(f[1])}')
        print(f'Test : {len(f[1])}')


    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")

    save_fold_csvs(dataset_for_folds, folds)
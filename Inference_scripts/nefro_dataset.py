from __future__ import print_function
from PIL import Image
import os
import os.path
import time
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import functional as F


'''
*random* training
mean: tensor([0.1224, 0.1224, 0.1224]) | std: tensor([0.0851, 0.0851, 0.0851])
'''

class Nefro(data.Dataset):
    """ Nefro Dataset. """

    data_root = 'c'
    filesdic = {
        'old_labels': data_root + "files/labels.csv",
        'labels': data_root + "files/whole_dataset_labels.csv",
        'images': data_root + "files/mtb_d_mdb.csv",
        'flags': data_root + "files/mdb.csv",
    }

    splitsdic = {
        'images': data_root + "files/whole_dataset_labels.csv",
        'files_root': data_root + "files/",
        'training': "_training.csv",
        'validation': "_validation.csv",
        'test': "_test.csv",
        'old_test': "_old_test.csv",

    }

    flagsdic = {
        'Lin': 2,
        'gran-gross': 3,
        'gran-fine': 4,
        'gen-segm': 5,
        'gen-diff': 6,
        'foc-segm': 7,
        'foc-glob': 8,
        'mesangiale': 9,
        'Par-regol-cont': 10,
        'Par-regol-discont': 11,
        'Par-irreg': 12,
        'Caps-Bow': 13,
        'parietale': 15,
        'parietal': 1,
    }

    def __init__(self, split_name='images', load=True, size=(224, 224), transform=None, label_name='mesangiale'):
        start_time = time.time()
        self.transform = transform
        self.load = load
        self.size = size

        print('loading ' + split_name)

        self.split_list, self.lbls = self.read_csv(split_name, label_name)

        if load:
            self.names, self.imgs = self.get_images(self.split_list, self.size)
            print(len(self.imgs))
    
        else:
            self.names = self.get_names(self.split_list)

        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: image
        """
        name = self.names[index]
        if not self.load:
            image = Image.open(self.names[index]).convert('I')
        else:
            image = self.imgs[index]

        if self.lbls[index] == 'True':
            label = 1
        else:
            label = 0

        if self.transform is not None:
            image = self.transform(image)

        return image, label, name

    def __len__(self):
        return len(self.split_list)

    @classmethod
    def get_names(cls, n_list):
        imgs = []

        for n in n_list:
            imgs.append(cls.data_root + 'images/' + n)

        return imgs

    @classmethod
    def get_images(cls, n_list, size):
        imgs = []

        for n in n_list:
            # im = Image.open(cls.data_root + 'images/' + n)
            # imarray = np.array(im)
            # imarray = np.divide(imarray, 16)
            # img = Image.fromarray(np.uint8(imarray)).convert('RGB')
            # imgs.append(img.resize(size, Image.BICUBIC))

            im = Image.open(cls.data_root + 'images/' + n).convert('I')
            imgs.append(im.resize(size, Image.BICUBIC))

        return n_list, imgs

    @classmethod
    def read_csv(cls, csv_filename, lblname):
        import csv
        split_list = []
        labels_list = []
        if csv_filename != 'images':
            fname = cls.splitsdic.get('files_root') + lblname + cls.splitsdic.get(csv_filename)
        else:
            fname = cls.splitsdic.get(csv_filename)
        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                split_list.append(row[0])
                labels_list.append(row[int(cls.flagsdic.get(lblname))])

        return split_list, labels_list


class NefroTiffToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x 1) in the range
    [0, 4095] to a torch.FloatTensor of shape (3 x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic = F.to_tensor(pic)
        # Alcune immagini erano a 12 bit quindi il massimo valore è 4095 e così normalizzo ogni valore del tensore in modo che sia fra 0 e 1
        pic = pic.float().div(4095)
        # Ripeto i canali 3 volte perchè Resnet prende in input immagini a 3 canali a meno di non retrainare il primo layer
        pic = pic.repeat(3, 1, 1)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

def denormalize(img):
    mean = 0.1224
    std = 0.0851
    for i in range(img.shape[0]):
        img[i, :, :] = img[i, :, :] * std
        img[i, :, :] = img[i, :, :] + mean
    return img


if __name__ == '__main__':
    
    nefro = Nefro()
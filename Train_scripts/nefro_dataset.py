from __future__ import print_function
from PIL import Image
import os
import os.path
import time
import csv
from collections import Counter
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
from torchvision.transforms import functional as F

'''
*random* training
mean: tensor([0.1224, 0.1224, 0.1224]) | std: tensor([0.0851, 0.0851, 0.0851])
'''


class Nefro(data.Dataset):
    """ Nefro Dataset. """

    data_root = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/'
    imgs_data_root = '/nas/softechict-nas-1/fpollastri/data/istologia/'
    img_rosati_data_root_3D = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv1/3DHISTECH'
    img_rosati_data_root_Hama = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv1/HAMAMATSU'



    # Aggiungo altre due colonne a questo dizionario che corrispondono alle colonne [Globale (GLOB) e Segmentale (SEGM)]
    flagsdic = {
        'LIN': 2,
        'PSEUDOLIN': 3,
        'GRAN_GROSS': 4,
        'GRAN_FINE': 5,
        'GEN_SEGM': 6,
        'GEN_DIFF': 7,
        'FOC_SEGM': 8,
        'FOC_GLOB': 9,
        'MESANGIALE': 10,
        'PARIETALE': 11,
        'PAR_REGOL_CONT': 12,
        'PAR_REGOL_DISCONT': 13,
        'PAR_IRREG': 14,
        'CAPS_BOW': 15,
        'POLOVASC': 16,
        'INTENS': 17,
        'GLOB': 18,
        'SEGM' : 19, 
    }

    def __init__(self, split, old_or_new_folder, w4k=False, wdiapo=False, label_name=[['MESANGIALE']], custom_name="", load=False,
                 size=(224, 224), transform=None):
        start_time = time.time()
        self.old_or_new_folder = old_or_new_folder
        self.splitsdict = {'files_root': self.data_root + old_or_new_folder}
        self.transform = transform
        self.load = load
        self.size = size
        
        self.tiff_transform = NefroTiffToTensor()
        self.diapo_transform = DiapoTiffToTensor()

        self.label_name = label_name
        self.classes = []
        # for c in label_name:
        #     self.classes.append([int(self.flagsdic.get(c))])

        for sublist in label_name:
            label = sublist[0]  # estrae 'PARIETALE_1'
            self.classes.append(int(self.flagsdic.get(label)))

        self.split_name = self.get_split_name(label_name, w4k, wdiapo, split, custom_name)
        print('loading ' + self.split_name)

        self.split_list, self.labels_list = self.read_dataset()

        self.class_distribution = Counter(self.labels_list)

        # self.split_list, self.lbls = self.read_csv(self.split_name, label_name)

        if load:
            self.names, self.imgs = self.get_images(self.split_list, self.size)
        else:
            self.names = self.get_names(self.split_list)
        
        print(f"[DEBUG] Lunghezza self.names: {len(self.names)}")
        print(f"[DEBUG] Lunghezza self.split_list: {len(self.split_list)}")
        print(f"[DEBUG] Lunghezza self.lbls: {len(self.lbls)}")
        print("Time: " + str(time.time() - start_time))

 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: image
        """

        # if self.lbls[index] == 'True' or self.lbls[index] == 'TRUE':
        #     label = 1
        # else:
        #     label = 0

        if not (0 <= index < len(self.split_list)):
            raise IndexError(f"Index {index} fuori range [0, {len(self.split_list)-1}]")

        # Debugging
        # print(f"[DEBUG] __getitem__ chiamato con index = {index}")
        # print(f"[DEBUG] self.split_list[{index}] = {self.split_list[index]}")
        # print(f"[DEBUG] self.lbls[{index}] = {self.lbls[index]}")
    

        label = self.lbls[index]
        name = self.names[index]
        if not self.load:
            if os.path.basename(os.path.dirname(name)) == 'diapo':
                image = Image.open(png_diapo(name))
            else:
                image = Image.open(name).convert('I')
        else:
            image = self.imgs[index]
    #TODO
    # Scambiare ordine tra augmentations e normalizzazione
    # Salvare tutte le immagini del batch nella enumerate del dataloader con Nick, cosi vedò cosa fanno le trasformazioni

        if self.transform is None:
            return np.asarray(image), label, name

        extension = os.path.splitext(name)[1].lower()

        image = self.transform(image)

        if extension == '.png':
            image = self.png_transform(image)
        elif extension == '.tif':
            image = self.tiff_transform(image)

        if os.path.basename(os.path.dirname(name)) == 'diapo':
            image = self.diapo_transform(image)

        return image, label, name

    def __len__(self):
        return len(self.split_list)

    def read_dataset(self):
        split_list = []
        labels_list = []
        fname = self.split_name

        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                try:
                    split_list.append(row[0])
                except:
                    print('Non ho trovato la riga corrispondente al nome di file : ', row)
                if self.label_name[0][0] == 'INTENS':
                    labels_list.append(float(row[self.flagsdic.get(self.label_name[0][0])]))
                else:
                    labels_list.append(get_single_label(row, self.classes))
        self.split_list = split_list
        self.lbls = labels_list
        print(f"[DEBUG] CSV caricato: {len(split_list)} nomi, {len(labels_list)} label")
        assert len(split_list) == len(labels_list), (
            f"Divergenza: split_list ({len(split_list)}) vs labels_list ({len(labels_list)})"
        )
        return split_list, labels_list

    @staticmethod
    def png_transform(img):
        """
        Prende un'immagine PNG, estrae il canale verde, lo replica 3 volte,
        lo converte in tensor e normalizza.
        """
        # Se l'immagine è ancora PIL.Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Debug
        #print(f"[DEBUG] Immagine shape: {img.shape}, dtype: {img.dtype}")  # <-- aggiunto anche dtype

        if img.ndim == 2:
            img_rgb = np.dstack([img, img, img])
        # If you want to use only Green:
        # elif img.ndim == 3:
        #     green = img[:, :, 1]
        #     img_rgb = np.dstack([green, green, green])
        
        # Cast esplicito a uint8 prima di trasformare
        img_rgb = img_rgb.astype(np.uint8)

        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img_rgb)

        # If you want to use only Green:
        # normalize = transforms.Normalize(
        #     mean=(0.1319403, 0.1319403, 0.1319403),
        #     std=(0.18537317, 0.18537317, 0.18537317)
        # )
        normalize = transforms.Normalize(
            mean=(0.13496943, 0.14678506, 0.13129657),
            std=(0.19465959, 0.19976119, 0.19709547)
        )

        img_normalized = normalize(img_tensor)

        return img_normalized


    @classmethod
    def get_names(cls, n_list):
        imgs = []
        # Qua devo inserire il controllo sul tipo di estensione delle mie immagini, perchè per ora non ho i permessi per copiare tutti i glomeruli nella cartella di Pollastri
        # Quindi, faccio il check per scegliere la data_root corretta
        for n in n_list:
            extension = os.path.splitext(n)[1]
            if extension == '.tif':
                imgs.append(cls.imgs_data_root + 'images/' + n)
            else:
                #print('Sto analizzando anche le immagini di Magistroni non solo quelle di Pollo')
                if n.startswith('R22'):
                    folder_name = n.split('_glomerulo')[0]
                    img_path = os.path.join(cls.img_rosati_data_root_3D, folder_name)
                    imgs.append(os.path.join(img_path, n))
                if n.startswith('R23'):
                    folder_name = n.split('_glomerulo')[0]
                    img_path = os.path.join(cls.img_rosati_data_root_Hama, folder_name)
                    imgs.append(os.path.join(img_path, n))
                if n.startswith('R24'):
                    folder_name = n.split('_glomerulo')[0]
                    img_path = os.path.join(cls.img_rosati_data_root_Hama, folder_name)
                    imgs.append(os.path.join(img_path, n))
                if n.startswith('R25'):
                    folder_name = n.split('_glomerulo')[0]
                    img_path = os.path.join(cls.img_rosati_data_root_Hama, folder_name)
                    imgs.append(os.path.join(img_path, n))
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
    def read_csv(cls, csv_filename, lblname, splitsdict):
        split_list = []
        labels_list = []
        if csv_filename != 'images':
            fname = splitsdict.get('files_root') + lblname + splitsdict.get(csv_filename)
        else:
            fname = splitsdict.get(csv_filename)
        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                split_list.append(row[0])
                labels_list.append(row[int(cls.flagsdic.get(lblname))])
        return split_list, labels_list


    def get_split_name(self, label, w4k, wdiapo, split, custom_name=''):
        final_name = str(label)
        if w4k:
            final_name += '_4k'
        if wdiapo:
            final_name += '_diapo'
        return self.splitsdict.get('files_root') + final_name + '_' + split + custom_name + '.csv'


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
        pic = pic.float().div(4095)
        pic = pic.repeat(3, 1, 1)
        pic = F.normalize(pic, (0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DiapoTiffToTensor(object):
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

        '''
        training
        mean: tensor([0.2090]) | std: tensor([0.1168])
        '''
        pic = pic[:, :, 1]
        pic = F.to_tensor(pic)
        pic = pic.float().div(255)
        pic = pic.repeat(3, 1, 1)
        # pic = F.normalize(pic, (0.2090, 0.2090, 0.2090), (0.1168, 0.1168, 0.1168))
        pic = F.normalize(pic, (0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851))

        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def denormalize(img):
    mean = 0.1224
    std = 0.0851
    # for i in range(img.shape[2]):
    #     img[:, :, i] = img[:, :, i] * std
    #     img[:, :, i] = img[:, :, i] + mean
    img *= std
    img += mean
    return torch.clamp(img, 0, 1)


def get_single_label(row, classes):
    for i, c in enumerate(classes):
        #for s_c in c:
        if row[c].upper() == 'TRUE':
            return i + 1
    return 0


def png_diapo(name):
    return name[:-4] + '.png'

    # for i, c in enumerate(classes):  # classes = [[0,1],[2]]
    #     for s_c in c:
    #         if row[s_c].upper() == 'TRUE':  # row ['img', 'False', 'True', 'False', ...]
    #             print(i + 1)
    #             break
    #     else:  # executed if the for does NOT break
    #         continue
    #     break  # executed if the for does BREAK
    # else:  # executed if the for does NOT break
    #     print(0)

if __name__ == '__main__':
            
    labels_to_use = [['MESANGIALE']]
    
    dataset = Nefro(
                split='training',
                old_or_new_folder = 'Files_old_Pollo/',
                w4k=True,
                wdiapo=False,
                label_name=labels_to_use,
                size=(512, 512),
                transform=None
    )
    
    class_distribution = dataset.class_distribution

    print('Questa è la distribuzione delle classi : ', class_distribution)

    print('Numero dati di train:', len(dataset))

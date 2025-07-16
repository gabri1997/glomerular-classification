from PIL import Image as PILImage
import numpy as np
from io import BytesIO
import imgaug.augmenters as iaa

def handle_input(inimg):

    if isinstance(inimg, PILImage.Image):
        img = inimg 
    else:
        img = PILImage.open(inimg)  

    imarray = np.array(img)  

    if img.mode == 'L': 
        pimarray = handle_grayscale(imarray)
    elif img.mode == 'RGB':  # Se l'immagine è a colori
        pimarray = handle_rgb(imarray)
    elif img.mode == 'I;16' and (img.tag.get(305) == ('analySIS 5.0',) or img.tag.get(271) == ('Olympus Soft Imaging Solutions',)):
        pimarray = handle_16bits(imarray)
    else:
        raise Exception('UNKNOWN DATA TYPE')

    pimarray = iaa.PadToFixedSize(width=max(pimarray.shape[0], pimarray.shape[1]),
                                  height=max(pimarray.shape[0], pimarray.shape[1]),
                                  pad_mode='constant', position='center').augment_image(pimarray)
    pimarray = iaa.Resize({"width": 512, "height": 512}).augment_image(pimarray)

    pimg = PILImage.fromarray(np.uint8(pimarray))  # Converte l'array numpy in un'immagine PIL
    buffer = BytesIO()
    pimg.save(fp=buffer, format='PNG')

    new_name = inimg.filename.split('.')[0] + '.png' if hasattr(inimg, 'filename') else 'processed_image.png'
    return buffer.getvalue(), new_name  # Restituisci i dati binari dell'immagine e il nome del file


def handle_grayscale(imarray):
    imarray = np.dstack((imarray, imarray, imarray))
    return imarray


def handle_rgb(imarray):
    # if not (imarray[:, :, 0] == imarray[:, :, 1]).all():
    #     # imarray = np.dstack((imarray[:, :, 1], imarray[:, :, 1], imarray[:, :, 1]))
    imarray = handle_grayscale(imarray[:, :, 1])
    return imarray


def handle_16bits(imarray):
    imarray = np.divide(imarray, 16)
    imarray = handle_grayscale(imarray)
    return imarray

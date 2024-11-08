import os
import glob
from PIL import Image
import numpy as np

# Funktion zum Laden der Bilder und Konvertieren in Tensoren
def load_images_from_folder(folder):
    dataset = []
    for filename in glob.glob(os.path.join(folder, '*.png')):  # oder *.jpg je nach Bildformat
        img = Image.open(filename).convert('L')  # In Graustufen umwandeln
        img = img.resize((3, 3))  # Falls nicht schon 3x3 Pixel
        img_array = np.array(img)
        dataset.append(img_array)

    return dataset

def split_images_to_sets(path_1: str, path_2: str):
    return load_images_from_folder(path_1), load_images_from_folder(path_2)
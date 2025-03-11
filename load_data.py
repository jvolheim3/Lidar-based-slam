import numpy as np
import os
import re
import numpy as np
from PIL import Image
import imageio
import pickle

dataset = 20

def load_images_from_folder(folder_path):
    image_files = os.listdir(folder_path)
    
    # Sort the image files based on the number at the end of their names
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x.split('.')[0])[-1]))

    images = []
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        img = imageio.imread(image_path) 
        # Convert the image to a numpy array
        img_array = np.array(img)
        images.append(img_array)
    return np.array(images)

folder_path = "../data/Disparity%d"%dataset
image_array = load_images_from_folder(folder_path)
print(image_array.shape)  # This will print the shape of the numpy array


with open('../data/Disparity%d.p'%dataset, 'wb') as file:
    pickle.dump(image_array, file)

def load_images_from_folder(folder_path):
    image_files = os.listdir(folder_path)
    
    # Sort the image files based on the number at the end of their names
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x.split('.')[0])[-1]))

    images = []
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        img = imageio.imread(image_path) 
        # Convert the image to a numpy array
        img_array = np.array(img)
        images.append(img_array)
    return np.array(images)

folder_path = "../data/RGB%d"%dataset
image_array = load_images_from_folder(folder_path)
print(image_array.shape)  # This will print the shape of the numpy array


with open('../data/RGB%d.p'%dataset, 'wb') as file:
    pickle.dump(image_array, file)

import os
from PIL import Image
import numpy as np

def yield_images(path_to_data):
    '''
    Function to yield images from a directory into a numpy generator
    path_to_data: path to the files
    '''
    for image in os.listdir(path_to_data):
        im = Image.open(os.path.join(path_to_data, image))
        np_im = np.array(im)
        yield(np_im)
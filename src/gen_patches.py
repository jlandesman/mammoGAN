#%matplotlib inline
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.transform import rotate
from skimage.util import view_as_windows
from collections import defaultdict

## Steps from https://arxiv.org/pdf/1707.06978.pdf:
## Resize calcifications to between 2750x1500 with random uniform sampling of the factors
## Resize masses to between  1100x600 with random uniform sampling of the factors
## Horizontal flipping 
## Rotation up to 30 degrees
## Use Otsu's segmentationt ot remove all purely black patches

np.random.seed(1234)

PATH_TO_FILES = '/home/jlandesman/data/cbis-ddsm/calc_training_full_mammogram_images/'
PATH_TO_ROI = '/home/jlandesman/data/cbis-ddsm/calc_training_full_roi_images/'
PATH_TO_ROI_CSV_LABELS = '/home/jlandesman/data/cbis-ddsm/calc_case_description_train_set.csv'

CALC_TARGET_RESIZE = np.array([2750,1500])
MASS_TARGET_RESIZE = np.array([1100, 600])
MAX_ROTATE = 30 ## degrees
STEP_SIZE = 40 ## Stride for getting windows
MASK_CUTOFF = 0 ## If a patch has an average mask value of 0 discard it as it is not in the breast
ROI_CUTOFF = 0 ## If an ROI has an average value of zero, label it "no_tumor" 

def get_im_as_array(file_name, file_type):
    '''
    Read in an image and yield it as a numpy array
    '''
    
    if file_type == 'full':
        path = PATH_TO_FILES
    else: ## ROI
        path = PATH_TO_ROI

    file_path = os.path.join(path,file_name)
    im = Image.open(file_path)
    return np.asarray(im)

def get_labels(path_to_csv):
    '''
    Concatenates various components of the named files to a list and returns the file_name and pathology
    params:
        path_to_csv: path to the CSV with the file list
    '''
    df = pd.read_csv(path_to_csv)
    df['file_name'] = 'Calc-Training_' + df['patient_id'] + '_' + df['side'] + '_' + df['view'] + '_' + df['abn_num'].astype(str) + '_mask.png'
    df = df[['file_name', 'pathology']]
    df.set_index('file_name', inplace=True)
    return df

def get_resize_max_min(im, tumor_type):
    '''
    Returns the max and min dimensions for resizing per the paper
    '''
    if tumor_type == 'CALC':
        resize_min, resize_max = CALC_TARGET_RESIZE/np.array(im.shape)
        
    else:
        resize_min, resize_max = MASS_TARGET_RESIZE/np.array(im.shape)
    
    return resize_min, resize_max                                                        

def get_resize_dims(im, dim_0, dim_1):
    '''
    Uniformly choose between resize_min and resize_max dimesnions 
    Default is the "second resizing" mentioned in the paper above
    Use random to determine if we want a new randomization or to preserve the last one
    or to make sure if the masks and images are identical
    '''
    if not_random:
        np.random.seed(1234)
    
    dim_0 = np.random.uniform(low = resize_min, high = resize_max)
    
    if not_random:
        np.random.seed(1234)
    
    dim_1 = np.random.uniform(low = resize_min, high = resize_max)
    return np.round([dim_0*im.shape[0], dim_1*im.shape[1]])

def rotate_image(im, rotation_angle):
    '''
    Rotates the image to a random angle < max_rotate
    '''
    return rotate(im, rotation_angle)

def normalize(im):
    '''
    Normalize to between 0 and 255
    '''
    im_normalized = (255*(im - np.max(im))/-np.ptp(im))    
    return im_normalized

def get_mask(im):
    '''
    Return OTSU's threshold mask. For use to make sure that the image patches are within the breast, not the empty space outside 
    '''
    ## OTSU's threshold
    thresh = threshold_otsu(im)
    binary = im >= thresh
    return(binary)

def get_patches(im, step_size = 20, dimensions = [256, 256]):
    '''
    Return sliding windows along the breast, moving 20 pixels at a time.
    '''
    patches = view_as_windows(im,dimensions,step=step_size)
    patches = patches.reshape([-1, 256, 256])
    return patches

def get_zipped_patches():
    return zip(get_patches(mammogram), get_patches(roi), get_patches(mask)) 

def get_mask_list():
    '''
    Associate each file with all of its masses and their pathology (benign, malignant, other).
    Return a dictionary of {file_name: (mask, pathology)}
    '''
    mask_list = defaultdict(list)
    roi_files = os.listdir(PATH_TO_ROI)
    df = get_labels(PATH_TO_ROI_CSV_LABELS)

    for file_name in roi_files:
        mask_list[file_name[:-11]].append((file_name, df.loc[file_name]['pathology']))
    
    return mask_list


def save_patches(zipped_patches, label, save_file_name):
    errors = []
    num_files = 0
    for number, patch in enumerate(zipped_patches):
        if patch[2].mean() == MASK_CUTOFF:
            continue ## Return to start of loop

        elif patch[1].mean() > 0: ## If this is in the tumor
            if label == 'MALIGNANT':
                save_path = '/home/jlandesman/data/patches/calcification/malignant'

            elif label == 'BENIGN':
                save_path = '/home/jlandesman/data/patches/calcification/benign'

            else:
                save_path = '/home/jlandesman/data/patches/calcification/benign_no_callback'

        else: ## Not in the tumor
            save_path = '/home/jlandesman/data/patches/calcification/no_tumor'

        file_name = save_file_name + "_" + str(number) + ".png"
        try:
            im = Image.fromarray(patch[0])
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(os.path.join(save_path, file_name))
            num_files += 1
        except:
            errors.append(file_name)
    print (num_files)
    print (len(errors))

    
###### RUN ###############
file_list = get_mask_list()

for mammogram_img in file_list.keys():
    print("New Image:", mammogram_img)
    mammogram = get_im_as_array(mammogram_img, 'full')
    
    mask = get_mask(mammogram)
    mask = mask/255 ## Normalize to between 0/1
    
    for roi_img in file_list[mammogram_img]:
        
        roi = get_im_as_array(roi_img[0], 'ROI')
        label = roi_img[1]
        
        zipped_patches = get_zipped_patches()
        save_patches(zipped_patches,label, mammogram_img)
        del(zipped_patches) ## Memory Management
    
    #################
    #Apply resize
    #################
    print("Starting Resize")
    resize_min, resize_max = get_resize_max_min(mammogram, 'CALC')
    
    dim_0 = np.random.uniform(low = resize_min, high = resize_max)
    dim_1 = np.random.uniform(low = resize_min, high = resize_max)
    
    resize_dims = np.round([dim_0*mammogram.shape[0], dim_1*mammogram.shape[1]])
    
    mammogram = resize(mammogram, resize_dims)
    mask = get_mask(mammogram)
    
    for roi_img in file_list[mammogram_img]:
        
        roi = get_im_as_array(roi_img[0], 'ROI')
        label = roi_img[1]
        roi = resize(roi, resize_dims)
        
        zipped_patches = get_zipped_patches()
        save_patches(zipped_patches,label, mammogram_img)
        del(zipped_patches) ## Memory Management

    #################
    #Apply rotation
    #################
    print("Starting Rotation")

    rotation_angle = np.random.randint(low = 0, high = MAX_ROTATE)
    mammogram = rotate_image(mammogram, rotation_angle)
    mask = get_mask(mammogram)
    
    for roi_img in file_list[mammogram_img]:
        
        roi = get_im_as_array(roi_img[0], 'ROI')
        label = roi_img[1]
        roi = rotate_image(roi, rotation_angle)
        
        zipped_patches = get_zipped_patches()
        save_patches(zipped_patches,label, mammogram_img)
        del(zipped_patches) ## Memory Management
    
    ######################
    #Apply Horizontal Flip
    #######################
    print("Starting Flip")

    mammogram = np.fliplr(mammogram)
    mask = get_mask(mammogram)
    
    for roi_img in file_list[mammogram_img]:
        
        roi = get_im_as_array(roi_img[0], 'ROI')
        label = roi_img[1]
        roi = np.fliplr(roi)
        
        zipped_patches = get_zipped_patches()
        save_patches(zipped_patches,label, mammogram_img)
        del(zipped_patches) ## Memory Management

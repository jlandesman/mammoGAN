from sklearn.model_selection import train_test_split
import os
import pandas as pd

TRAIN_PATH = '/home/jlandesman/data/patches/calcification/train/'
TEST_PATH = '/home/jlandesman/data/patches/calcification/test/'

## Read in files
no_tumor = os.listdir(TRAIN_PATH + 'no_tumor')
benign = os.listdir(TRAIN_PATH + 'benign')
benign_no_callback = os.listdir(TRAIN_PATH+'benign_no_callback')
malignant = os.listdir(TRAIN_PATH + 'malignant')

## Build DF
file_paths = no_tumor + benign + benign_no_callback + malignant
labels = ['no_tumor'] * len(no_tumor) + ['benign'] * len(benign) + ['benign_no_callback'] * len(benign_no_callback) + ['malignant'] * len(malignant)
assert len(file_paths) == len(labels)

df = pd.DataFrame({'file_paths': file_paths, 'labels': labels})

## Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(df['file_paths'], df['labels'], test_size = 0.2, random_state = 142)

## Run
counter = 0
for label, file_name in zip(Y_test, X_test):
    current_dir = os.path.join(TRAIN_PATH + label +'/'+ file_name)
    test_dir = os.path.join(TEST_PATH + label +'/'+ file_name)
    os.rename(current_dir, test_dir)
    counter += 1
    if counter%1000 == 0:
        print ('Files moved; ', counter)
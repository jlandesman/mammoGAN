#!bin/bash

mkdir data
mkdir data/cbis_training
mkdir data/mass_training_cropped
mkdir data/mass_training_full

gcsfuse cbis-ddsm-calc-training-full-mammogram-images /home/jlandesman/data/cbis_training

gcsfuse cbis-ddsm-mass-trainng-roi-and-cropped-images /home/jlandesman/data/mass_training_cropped

gcsfuse cbis-ddsm-mass-training-full-mammogram-images /home/jlandesman/data/mass_training_full


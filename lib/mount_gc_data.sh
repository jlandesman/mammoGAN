#!bin/bash
sudo rm -rf data/
mkdir data
mkdir data/train
mkdir data/train/mass_cropped
mkdir data/train/mass_full
mkdir data/train/calc_full
mkdir data/train/calc_cropped

mkdir data/test
mkdir data/test/mass_cropped
mkdir data/test/mass_full
mkdir data/test/calc_full
mkdir data/test/calc_cropped

mkdir data/mias_mini
chmod -R 777 data

gcsfuse cbis-ddsm-calc-training-full-mammogram-images data/train/calc_full
gcsfuse cbis-ddsm-calc-training-roi-and-cropped-images data/train/calc_cropped
gcsfuse cbis-ddsm-mass-training-roi-and-cropped-images data/train/mass_cropped
gcsfuse cbis-ddsm-mass-training-full-mammogram-images data/train/mass_full

gcsfuse cbis-ddsm-calc-test-full-mammogram-images data/test/calc_full
gcsfuse cbis-ddsm-calc-test-roi-and-cropped-images data/test/calc_cropped
gcsfuse cbis-ddsm-mass-test-roi-and-cropped-images data/test/mass_cropped
gcsfuse cbis-ddsm-mass-test-full-mammogram-images data/test/mass_full

gcsfuse mias-mini-png data/mias_mini



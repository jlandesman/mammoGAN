#!bin/bash
sudo rm -rf data/
mkdir data
mkdir data/training
mkdir data/training/mass_cropped
mkdir data/training/mass_full
mkdir data/training/calc_full
mkdir data/training/calc_cropped

mkdir data/test
mkdir data/test/mass_cropped
mkdir data/test/mass_full
mkdir data/test/calc_full
mkdir data/test/calc_cropped

mkdir data/mias_mini
chmod -R 777 data

gcsfuse cbis-ddsm-calc-training-full-mammogram-images /home/jlandesman/data/training/calc_full
gcsfuse cbis-ddsm-calc-training-roi-and-cropped-images /home/jlandesman/data/training/calc_cropped
gcsfuse cbis-ddsm-mass-training-roi-and-cropped-images /home/jlandesman/data/training/mass_cropped
gcsfuse cbis-ddsm-mass-training-full-mammogram-images /home/jlandesman/data/training/mass_full

gcsfuse cbis-ddsm-calc-test-full-mammogram-images /home/jlandesman/data/test/calc_full
gcsfuse cbis-ddsm-calc-test-roi-and-cropped-images /home/jlandesman/data/test/calc_cropped
gcsfuse cbis-ddsm-mass-test-roi-and-cropped-images /home/jlandesman/data/test/mass_cropped
gcsfuse cbis-ddsm-mass-test-full-mammogram-images /home/jlandesman/data/test/mass_full

gcsfuse mias-mini-tiff /home/jlandesman/data/mias_mini



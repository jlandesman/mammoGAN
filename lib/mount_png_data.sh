#!bin/bash
sudo rm -rf data_png/
mkdir data_png

mkdir data_png/mass_cropped
mkdir data_png/mass_full

mkdir data_png/calc_full
mkdir data_png/calc_cropped

mkdir data_png/mias_mini
mkdir data_png/mias_mini_balanced

chmod -R 777 data_png

gcsfuse --implicit-dirs cbis-ddsm-calc-full-mammogram-images data_png/calc_full
gcsfuse --implicit-dirs cbis-ddsm-calc-roi-and-cropped-images data_png/calc_cropped
gcsfuse --implicit-dirs cbis-ddsm-mass-roi-and-croppsed-images data_png/mass_cropped
gcsfuse --implicit-dirs cbis-ddsm-mass-full-mammogram-images data_png/mass_full
gcsfuse --implicit-dirs mias-mini data_png/mias_mini
gcsfuse --implicit-dirs mias-mini-balanced data_png/mias-mini-balanced

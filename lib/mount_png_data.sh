#!bin/bash
sudo rm -rf data_png/
mkdir data_png
mkdir data_png/mass_cropped
mkdir data_png/mass_full

mkdir data_png/calc_full
mkdir data_png/calc_cropped

mkdir data_png/mias_mini

chmod -R 777 data_png

gcsfuse cbis-ddsm-calc-full-mammogram-images data_png/calc_full
gcsfuse cbis-ddsm-calc-roi-and-cropped-images data_png/calc_cropped
gcsfuse cbis-ddsm-mass-roi-and-croppsed-images data_png/mass_cropped
gcsfuse cbis-ddsm-mass-full-mammogram-images data_png/mass_full
gcsfuse mias-mini-png data_png/mias_mini

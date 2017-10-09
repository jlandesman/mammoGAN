#!bin/bash
sudo rm -rf data
mkdir data
mkdir data/training
mkdir data/test
mkdir data/mias_mini

gcsfuse cbis-ddsm-calc-training-full-mammogram-images /home/jlandesman/data/training
gcsfuse cbis-ddsm-calc-training-roi-and-cropped-images /home/jlandesman/data/trainnig
gcsfuse cbis-ddsm-mass-training-roi-and-cropped-images /home/jlandesman/data/training
gcsfuse cbis-ddsm-mass-training-full-mammogram-images /home/jlandesman/data/training

gcsfuse cbis-ddsm-calc-test-full-mammogram-images /home/jlandesman/data/training
gcsfuse cbis-ddsm-calc-test-roi-and-cropped-images /home/jlandesman/data/trainnig
gcsfuse cbis-ddsm-mass-test-roi-and-cropped-images /home/jlandesman/data/training
gcsfuse cbis-ddsm-mass-test-full-mammogram-images /home/jlandesman/data/training

gcsfuse mias-mini-tiff /home/jlandesman/data/mias_mini



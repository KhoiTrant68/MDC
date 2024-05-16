#!/bin/bash

# Install the required Python packages
pip install -r requirements.txt

# Download imagenet1000 dataset from kaggle to dataset
#cd datasets
#pip install --user kaggle
#mkdir ~/.kaggle
#mv kaggle.json ~/.kaggle
#chmod 600 ~/.kaggle/kaggle.json
#kaggle datasets download -d ambityga/imagenet100

# Unzip and reorganize the dataset
unzip archive.zip -d imagenet100
rm -rf archive.zip
python reconstruct_dataset.py imagenet100 imagenet
cd ..

# Download pretrained weights for MAE
wget -nc -P ./pretrained_models https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
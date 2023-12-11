#!/bin/bash

# Install the required Python packages
pip install -r requirements.txt

# Git clone CompressAI
!git clone https://github.com/InterDigitalInc/CompressAI.git
cd CompressAI
pip install -e .
cd ..

# Download pretrained model
wget -nc -P ./checkpoint https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

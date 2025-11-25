# Overview

This is the PyTorch implementation of paper “BeamCKM: A Framework of Channel Knowledge Map Construction for Multi-Antenna Systems”.

# requirements

To use this project, you can run the following command.

```cmd
conda create -n yourenv python=3.11
conda activate yourenv
pip install -r ./requirement/requirements.txt
```

# Project Preparation

### A. Data Preparation

The multi-beam channel knowledge map dataset is generated from [Sionna Toolbox](https://nvlabs.github.io/sionna/). The environmental contour is obtained from the [openstreetmap](https://www.openstreetmap.org/) website. This paper provides a trainable dataset in [Google Drive](https://drive.google.com/drive/folders/1rXx10-FE3ALH-57TAh9_2ltZEt0JnjDk), which is easier to use for multi-antenna channel knowledge map construction.

You can generate your own dataset according to the [open source library of Sionna](https://nvlabs.github.io/sionna/) as well. The details of data pre-processing can be found in the website.

The pretrained model to be loaded can be downloaded from [Google Drive](https://drive.google.com/file/d/19Otp9t7ne8iqazO4Bd4GBef9_UqIiwIX/view?usp=drive_link).

### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
├── 📁 BeamCKMSeer
│   ├── 📁 csv
│   ├── 📁 data
│   └── 📁 png
├── 📁 dataLoader
│   ├── 📄 __init__.py
│   └── 📄 loader.py
├── 📁 lossFunction
│   ├── 📄 __init__.py
│   └── 📄 loss.py
├── 📁 model
    └── 📁 
├── 📁 networks
│   ├── 📄 utils.py
│   ├── 📄 vit_seg_configs.py
│   ├── 📄 vit_seg_modeling.py
│   └── 📄 vit_seg_modeling_resnet_skip.py
├── 📁 pretrained
│   └── 📄 R50+ViT-B_16.npz
├── 📁 requirement
│   └── 📄 requirements.txt
├── 📄 train.py
├── 📄 trainer.py
└── 📄 tree.py
```

# Train CKMTransUNet 

An example of run.sh is listed below. Simply use it with `sh run.sh`. It will start advanced scheme aided training.

```cmd
python3 train.py \
--dataset 'BeamCKM' \ 
--batch_size 16 \
--max_epochs 50 \
--base_lr 0.0001
```

You can also use accelerate toolbox to reduce the GPU memory and accelerate training.

```cmd
accelerate launch --num_processes=1 --main_process_port=0 train.py \
--dataset 'BeamCKM' \ 
--batch_size 16 \
--max_epochs 50 \
--base_lr 0.0001
```

**PS：**If different beam numbers are considered, the user need to modify some codes. The code requiring modification is indicated by comments in the python files.

# Results and Reproduction

The main results reported in our paper are presented as follows. 

------

<img src="./src/image1.png" alt="image-20251124112727059" style="zoom:80%;" />

------

<img src="./src/image2.png" alt="image-20251124112727059" style="zoom:80%;" />

------



# Acknowledgement

This code is based on the [TransUNet](https://github.com/Beckschen/TransUNet) repository. We thank the authors for their valuable work.
# MS-RAFT-3D

## Installation

### Packages
We use CUDA version 11.3. and PyTorch version 1.12.1 installing the packages
```
numpy
torch
torchvision
scikit-sparse
Pillow
opencv-python
matplotlib
tqdm
```

Additionally we install
```bash
pip install git+https://github.com/jasc7636/lietorch.git@b59f04a63f2a5c8205191043659c14139588d578

git clone git@github.com:cv-stuttgart/flow_library.git
cd flow_library
pip install -r requirements.txt
```
and add the ```flow_library``` path to the ```PYTHONPATH``` environment variable.

### On-Demand Cost Volume
In order to save GPU VRAM, the on-demand cost volume can be used. Running
```
python setup.py install
```
in the ```alt_cuda_corr``` folder installs it.

## Datasets
The dataset location can be set with the environment variables ```DATASETS_SCENEFLOW_ROOT```, ```DATASETS_KITTI_ROOT```, ```DATASETS_KITTI_DISP_TRAIN```, ```DATASETS_KITTI_DISP_TEST```, ```DATASETS_SPRING_ROOT``` and ```DATASETS_SPRING_DISP```. The expected folder layout is the following:
```
DATASETS_SCENEFLOW_ROOT
├── fth
    ├── frames_cleanpass
    ├── frames_finalpass
    ├── disparity
    ├── disparity_change
    ├── optical_flow
    ├── camera_data


DATASETS_KITTI_ROOT
├── testing
    ├── calib_cam_to_cam
    ├── image_2
    ├── image_3
├── training
    ├── calib_cam_to_cam
    ├── image_2
    ├── image_3
    ├── disp_occ_0
    ├── disp_occ_1
    ├── flow_occ

DATASETS_KITTI_DISP_TRAIN
├── 000000_10.png
├── 000000_11.png
├── 000001_10.png
├── 000001_11.png
├── ...

DATASETS_KITTI_DISP_TEST
├── 000000_10.png
├── 000000_11.png
├── 000001_10.png
├── 000001_11.png
├── ...


DATASETS_SPRING_ROOT
├── test
    ├── 0003
        ├── cam_data
        ├── frame_left
        ├── frame_right
        ├── disp1_left
        ├── disp1_right
        ├── disp2_FW_left
        ├── disp2_FW_right
        ├── disp2_BW_left
        ├── disp2_BW_right
        ├── flow_FW_left
        ├── flow_FW_right
        ├── flow_BW_left
        ├── flow_BW_right
    ├── ...
├── train
    ├── 0001
        ├── cam_data
        ├── frame_left
        ├── frame_right
    ├── ...

DATASETS_SPRING_DISP
├── test
    ├── 0003
        ├── disp1_left
        ├── disp1_right
    ├── ...
├── train
    ├── 0001
        ├── disp1_left
        ├── disp1_right
    ├── ...
```

For evaluation on the FlyingThings3D dataset, the test file from RAFT-3D [things_test_data.pickle](https://drive.google.com/file/d/1zzPAJ-hYlA0eKgzwwuuh3zfS47OXD7su/view?usp=sharing) is required.
The environment variable ```DATASETS_SCENEFLOW_TEST_FILE``` should contain the path to this file.


## Usage

The network settings and training process is controlled with a JSON file.
Examples can be found in the ```config``` directory.

#### Training
```bash
# Pretraining
python3 scripts/train.py --config=config/3-scale_things.json --save=/checkpoint/folder

# Fine-tuning
python3 scripts/train.py --config=config/3-scale_kitti.json --save=/checkpoint/folder --ckpt=/path/to/checkpoint
python3 scripts/train.py --config=config/3-scale_spring.json --save=/checkpoint/folder --ckpt=/path/to/checkpoint
```

#### Evaluation
```bash
python3 scripts/evaluation.py --config=config/3-scale_things.json --model=/path/to/checkpoint --dataset="kitti+sceneflow+spring"
```

#### Submission
```bash
# KITTI
python3 scripts/kitti_submission.py --config=config/3-scale_things.json --model=/path/to/checkpoint

# Spring
python3 scripts/spring_submission.py --config=config/3-scale_things.json --model=/path/to/checkpoint
```

## Acknowledgement
Parts of the repository are adapted from [RAFT-3D](https://github.com/princeton-vl/RAFT-3D/) and [CCMR](https://github.com/cv-stuttgart/CCMR). We thank the authors.
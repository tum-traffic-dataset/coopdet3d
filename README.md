# CoopDet3D: Deep Multi-Modal Cooperative 3D Object Detection of Traffic Participants Using Onboard and Roadside Sensors
<div align="center">
<a href="https://tum-traffic-dataset.github.io/tumtraf-v2x"><img src="https://img.shields.io/badge/Website-CoopDet3D-0065BD.svg" alt="Website Badge"/></a>
<a href="https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4"><img src="https://img.shields.io/badge/Dataset-TUMTraf_V2X-0065BD.svg?style=flat&logo=github&logoColor=white" alt="Github Badge"/></a>
<a href="https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit"><img src="https://img.shields.io/badge/Dev_Kit-code-E37222.svg?style=flat&logo=github&logoColor=white" alt="Github Badge"/></a>
<a href="https://arxiv.org/pdf/2403.01316.pdf"><img src="https://img.shields.io/badge/Paper-TUMTraf_V2X-a2ad00.svg" alt="Paper Badge"/></a>

<a href="https://github.com/walzimmer/3d-bat"><img src="https://img.shields.io/badge/Labeling_Tool-3D_BAT-BC4E99.svg?style=flat&logo=github" alt="Paper Badge"/></a>
<a href="https://ieeexplore.ieee.org/document/8814071"><img src="https://img.shields.io/badge/Paper-3D_BAT-0E2B3A.svg?style=flat&logo=ieee" alt="Paper Badge"/></a>
<a href="https://github.com/walzimmer/3d-bat/stargazers"><img src="https://img.shields.io/github/stars/tum-traffic-dataset/coopdet3d" alt="Stars Badge"/></a>
<a href="https://github.com/walzimmer/3d-bat/network/members"><img src="https://img.shields.io/github/forks/tum-traffic-dataset/coopdet3d" alt="Forks Badge"/></a>
<a href="https://github.com/walzimmer/3d-bat/pulls"><img src="https://img.shields.io/github/issues-pr/tum-traffic-dataset/coopdet3d" alt="Pull Requests Badge"/></a>
<a href="https://github.com/walzimmer/3d-bat/issues"><img src="https://img.shields.io/github/issues/tum-traffic-dataset/coopdet3d" alt="Issues Badge"/></a>
<a href="https://a9-dataset.innovation-mobility.com/license"><img src="https://img.shields.io/badge/license-custom-2b9348.svg" alt="Issues Badge"/></a>
</div>

<p align="center">
<img src="imgs/coopdet3d_architecture_v2.png" width="1000" alt="" class="img-responsive">
</p>

## Abstract

Cooperative perception offers several benefits for enhancing the capabilities of autonomous vehicles and improving road safety. Using roadside sensors in addition to onboard sensors increases reliability and extends the sensor range. External sensors offer higher situational awareness for automated vehicles and prevent occlusions. We propose CoopDet3D, a cooperative multi-modal fusion model, and TUMTraf-V2X, a dataset for the cooperative 3D object detection and tracking task. Our dataset contains 2,000 labeled point clouds and 5,000 labeled images from five roadside and four onboard sensors. It includes 30k 3D boxes with track IDs and precise GPS and IMU data. We labeled nine categories and covered occlusion scenarios with challenging driving maneuvers, like traffic violations, near-miss events, overtaking, and U-turns. Through multiple experiments, we show that our CoopDet3D camera-LiDAR fusion model achieves an increase of +14.36 3D mAP compared to a vehicle camera-LiDAR fusion model. Finally, we make our dataset, model, labeling tool, and dev-kit publicly available: https://tum-traffic-dataset.github.io/tumtraf-v2x.

## Overview ✨
- [News](#news)
- [Features](#features)
- [Dataset Download](#dataset-download)
- [Pretrained Weights](#pretrained-weights)
- [Usage](#usage)
- [Benchmark](#benchmark)
- [Acknowledgment](#acknowledgement)

## News 📢
- 2024/02: 🏆 Accepted paper at [CVPR'24](https://cvpr.thecvf.com/) conference: [TUMTraf V2X Cooperative Perception Dataset](https://arxiv.org/pdf/2403.01316.pdf)
- 2023/11: First release of the CoopDet3D model (v1.0.0)

## Features 🔥
- Support vehicle-only, infrastructure-only, and cooperative modes
    - [x] Vehicle-only
    - [x] Infrastructure-only
    - [x] Cooperative
- Support camera-only, LiDAR-only, and camera-LiDAR fusion
    - [x] Camera-only
    - [x] LiDAR-only
    - [x] Camera-LiDAR fusion
- Support multiple camera backbones
    - [x] SwinT
    - [x] YOLOv8
- Support multiple LiDAR backbones
    - [x] VoxelNet ([torchsparse](https://github.com/mit-han-lab/torchsparse))
    - [x] PointPillars
- Support offline, ROS, and shared memory operation
    - [x] Offline
    - [ ] ROS
    - [ ] Shared memory
    - [ ] Live Test
- Export inference results to OpenLABEL format
    - [x] Inference to OpenLABEL

## Dataset Download 📂

1. There are two versions of the [TUMTraf V2X Cooperative Perception Dataset](https://arxiv.org/pdf/2403.01316.pdf) (Release R4) provided:

    1.1. [TUMTraf-V2X](https://innovation-mobility.com/tumtraf-dataset)

    1.2. [TUMTraf-V2X-mini](https://innovation-mobility.com/tumtraf-dataset) (half of the full dataset)

We train CoopDet3D on TUMTraf-V2X-mini and provide the results below.

Simply place the splits in a directory named `tumtraf_v2x_cooperative_perception_dataset` in the `data` directory and you should have a structure similar to this:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_v2x_cooperative_perception_dataset
|   |   ├── train
|   |   ├── val
```

2. The [TUMTraf Intersection Dataset](https://ieeexplore.ieee.org/document/10422289) (Release R2) can be downloaded below:

    2.1 [TUMTraf-I](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_2).

Then, download the [TUMTraf Dataset Development Kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit) and follow the steps provided there to split the data into train and val sets.

Finally, place the train and val sets in a directory named `tumtraf_i` in the `data` directory. You should then have a structure similar to this:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_i
|   |   ├── train
|   |   ├── val
```



## Pretrained Weights 🚀

The pre-trained CoopDet3D weights can be downloaded from [here](https://nx21496.your-storageshare.de/s/MYW3dSJz7sCwRbH).

The weights for TUMTraf Intersection Dataset are named following this convention: 
```
coopdet3d_tumtraf_i_[l/cl]_<LiDAR_backbone>_<camera_backbone>_<other_information>.pth
```
The weights for the TUMTraf V2X Cooperative Perception Dataset are named following this convention:
```
coopdet3d_[v/i/vi]_[c/l/cl]_<LiDAR_backbone>_<camera_backbone>_<other_information>.pth
```
Extract the files and place them in the `weights` directory.

Use these weights to get the best results from the tables below:
- TUMTraf Intersection Dataset:
[coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth](https://nx21496.your-storageshare.de/s/28HAwnT6RZg3XQ7)
- TUMTraf V2X Cooperative Perception Dataset: [coopdet3d_vi_cl_pointpillars512_2xtestgrid_yolos_transfer_learning_best.pth](https://nx21496.your-storageshare.de/s/HiLBkE5WGeFKYSm)

## Usage 🌟

### Working with Docker

The easiest way to deal with prerequisites is to use the [Dockerfile](docker/Dockerfile) included. Make sure that `nvidia-docker` is installed on your machine. After that, execute the following command to build the docker image:

```bash
cd docker && docker build . -t coopdet3d
```

The docker can then be run with the following commands:

If you are only using the TUMTraf Intersection Dataset:

```bash
nvidia-docker run -it -v `pwd`/../data/tumtraf_i:/home/data/tumtraf_i -v <PATH_TO_COOPDET3D>:/home/coopdet3d --shm-size 16g coopdet3d /bin/bash
```

If you are only using the TUMTraf V2X Cooperative Perception Dataset:

```bash
nvidia-docker run -it -v `pwd`/../data/tumtraf_v2x_cooperative_perception_dataset:/home/data/tumtraf_v2x_cooperative_perception_dataset -v <PATH_TO_COOPDET3D>:/home/coopdet3d --shm-size 16g coopdet3d /bin/bash
```

If you are using both datasets:

```bash
nvidia-docker run -it  -v `pwd`/../data/tumtraf_i:/home/data/tumtraf_i -v `pwd`/../data/tumtraf_v2x_cooperative_perception_dataset:/home/data/tumtraf_v2x_cooperative_perception_dataset -v <PATH_TO_COOPDET3D>:/home/coopdet3d --shm-size 16g coopdet3d /bin/bash
```

It is recommended for users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Inside the docker, run the following command to install the codebase:

```bash
cd /home/coopdet3d
python setup.py develop
```

Finally, you can create a symbolic link `/home/coopdet3d/data/tumtraf_i` to `/home/data/tumtraf_i` and `/home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset` to `/home/data/tumtraf_v2x_cooperative_perception_dataset` in the docker.

### Working without Docker

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)
- Latest versions of numba, [torchsparse](https://github.com/mit-han-lab/torchsparse), pypcd, and Open3D

After installing these dependencies, run this command to install the codebase:

```bash
python setup.py develop
```

Finally, you can create a symbolic link `/home/coopdet3d/data/tumtraf_i` to `/home/data/tumtraf_i` and `/home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset` to `/home/data/tumtraf_v2x_cooperative_perception_dataset` in the docker.

### Data Preparation

#### TUMTraf Intersection Dataset

Run this script for data preparation:

```bash
python ./tools/create_tumtraf_data.py --root-path /home/coopdet3d/data/tumtraf_i --out-dir /home/coopdet3d/data/tumtraf_i_processed --splits training,validation
```

After data preparation, you will be able to see the following directory structure:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_i
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_i_processed
│   │   ├── tumtraf_nusc_gt_database
|   |   ├── train
|   |   ├── val
│   │   ├── tumtraf_nusc_infos_train.pkl
│   │   ├── tumtraf_nusc_infos_val.pkl
│   │   ├── tumtraf_nusc_dbinfos_train.pkl

```

#### TUMTraf V2X Cooperative Perception Dataset

Run this script for data preparation:

```bash
python ./tools/create_tumtraf_v2x_data.py --root-path /home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset --out-dir /home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset_processed --splits training,validation
```

After data preparation, you will be able to see the following directory structure:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_v2x_cooperative_perception_dataset
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_v2x_cooperative_perception_dataset_processed
│   │   ├── tumtraf_v2x_nusc_gt_database
|   |   ├── train
|   |   ├── val
│   │   ├── tumtraf_v2x_nusc_infos_train.pkl
│   │   ├── tumtraf_v2x_nusc_infos_val.pkl
│   │   ├── tumtraf_v2x_nusc_dbinfos_train.pkl

```

### Training

**NOTE 1:** If you want to use a YOLOv8 `.pth` file from MMYOLO, please make sure the keys inside fit with this model. Convert that `.pth` checkpoint using this converter: `./tools/convert_yolo_checkpoint.py`. 

**Note 2:** The paths to the pre-trained weights for YOLOv8 models are hardcoded in the config file, so change it there accordingly. This also means that when training models that use YOLOv8, the parameters `--model.encoders.camera.backbone.init_cfg.checkpoint`, `--model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint`, and `--model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint` are optional.

**Note 3:** We trained our model on 3 GPUs (3 x RTX 3090) and used the following prefix for that: `torchpack dist-run -np 3` 

For training a camera-only model on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE> --model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> 
```

Example:

```
torchpack dist-run -np 3 python tools/train.py configs/tumtraf_i/det/centerhead/lssfpn/camera/256x704/yolov8/default.yaml
```

For training LiDAR-only model on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE>
```

Example:

```
torchpack dist-run -np 3 python tools/train.py configs/tumtraf_i/det/transfusion/secfpn/lidar/pointpillars.yaml
```

For training a fusion model on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE> --model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> --load_from <PATH_TO_PRETRAINED_LIDAR_PTH>
```

Example:

```
torchpack dist-run -np 3 python tools/train.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --load_from weights/coopdet3d_tumtraf_i_l_pointpillars512_2x.pth
```

For training camera-only model on the TUMTraf V2X Cooperative Perception Dataset, run:

```
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE> --model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> --model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> 
```

Use the pretrained camera parameters depending on which type of model you want to train: vehicle-only, camera-only, or cooperative (both).


Example:

```
torchpack dist-run -np 3 python tools/train_coop.py configs/tumtraf_v2x/det/centerhead/lssfpn/cooperative/camera/256x704/yolov8/default.yaml
```

For training LiDAR-only model on the TUMTraf V2X Cooperative Perception Dataset, run:

```
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE>
```

Example:

```
torchpack dist-run -np 3 python tools/train_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/lidar/pointpillars.yaml
```

For training fusion model on the TUMTraf V2X Cooperative Perception Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE> ---model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --load_from <PATH_TO_PRETRAINED_LIDAR_PTH>
```
Use the pretrained camera parameters depending on which type of model you want to train: vehicle-only, camera-only, or cooperative (both).

Example:

```
torchpack dist-run -np 3 python tools/train_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml --load_from weights/coopdet3d_vi_l_pointpillars512_2x.pth
```

Note: please run `tools/test.py` or `tools/test_coop.py` separately after training to get the final evaluation metrics.

### BEV mAP Evaluation (Customized nuScenes Protocol)

**NOTE: This section will not work without the test set ground truth, which is not made public. To evaluate your model's mAP<sub>BEV</sub>, please send your config files and weights to the authors for evaluation!**

For evaluation on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 1 python tools/test.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --eval bbox
```

Example:

```
torchpack dist-run -np 1 python tools/test.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --eval bbox
```

For evaluation on the TUMTraf V2X Cooperative Perception Dataset, run:

```
torchpack dist-run -np 1 python tools/test_coop.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --eval bbox
```

Example:

```
torchpack dist-run -np 1 python tools/test_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_vi_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --eval bbox
```

### Running CoopDet3D Inference and Save Detections in OpenLABEL Format

Exporting to OpenLABEL format is needed to perform mAP<sub>3D</sub> evaluation or detection visualization using the scripts in the [TUM Traffic dev-kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit).

**NOTE: You will not be evaluate your inference results using the dev-kit without the test set ground truth, which is not made public. To evaluate your model's mAP<sub>3D</sub>, please send your detection results to the authors for evaluation!**

For TUMTraf Intersection Dataset:

```
torchpack dist-run -np 1 python tools/inference_to_openlabel.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --out-dir <PATH_TO_OPENLABEL_OUTPUT_FOLDER>
```

Example:

```
torchpack dist-run -np 1 python tools/inference_to_openlabel.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --split test --out-dir inference
```

For TUMTraf V2X Cooperative Perception Dataset:

```
torchpack dist-run -np 1 python scripts/cooperative_multimodal_3d_detection.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_CHECKPOINT_PTH> --split [train, val, test] --input_type hard_drive --save_detections_openlabel --output_folder_path_detections <PATH_TO_OPENLABEL_OUTPUT_FOLDER>
```

Example:
```
torchpack dist-run -np 1 python scripts/cooperative_multimodal_3d_detection.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/bevfusion_coop_vi_cl_pointpillars512_2x_yolos.pth --split test --input_type hard_drive --save_detections_openlabel --output_folder_path_detections inference
```

### Runtime Evaluation:

For TUMTraf Intersection Dataset:

```
torchpack dist-run -np 1 python tools/benchmark.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --log-interval 50
```

Example:

```
torchpack dist-run -np 1 python tools/benchmark.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --log-interval 50
```

For TUMTraf V2X Cooperative Perception Dataset:

```
torchpack dist-run -np 1 python tools/benchmark_coop.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --log-interval 10
```

Example:

```
torchpack dist-run -np 1 python tools/benchmark_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_vi_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --log-interval 10
```

### Built in visualization:

For TUMTraf Intersection Dataset:

```
torchpack dist-run -np 1 python tools/visualize.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --mode pred --out-dir viz_tumtraf 
```

Example:

```
torchpack dist-run -np 1 python tools/visualize.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --split test --mode pred --out-dir viz_tumtraf 
```

For TUMTraf V2X Cooperative Perception Dataset:

```
torchpack dist-run -np 1 python tools/visualize_coop.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --mode pred --out-dir viz_tumtraf 
```

Example:

```
torchpack dist-run -np 1 python tools/visualize_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/coopdet3d_vi_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --split test --mode pred --out-dir viz_tumtraf 
```

For split, naturally one could also choose "train" or "val". For mode, the other options are "gt" (ground truth) or "combo" (prediction and ground truth).

**NOTE: Ground truth visualization on test set will not work since the test set provided is missing the ground truth.**

## Benchmark 🎯

### Evaluation Results (mAP<sub>BEV</sub> and mAP<sub>3D</sub> ) of CoopDet3D on TUMTraf V2X Cooperative Perception Dataset Test Set in South 2 FOV

| Domain        | Modality    | mAP<sub>BEV</sub> | mAP<sub>3D</sub> Easy | mAP<sub>3D</sub> Mod. | mAP<sub>3D</sub> Hard | mAP<sub>3D</sub> Avg. |
|--------------|-------------|----------------|----------------|--------------|--------------|--------------|
| Vehicle | Camera | 46.83 | 31.47 | 37.82 | 30.77 | 30.36 |
| Vehicle | LiDAR | 85.33 | 85.22 | 76.86 | 69.04 | 80.11 |
| Vehicle | Cam+LiDAR | 84.90 | 77.60 | 72.08 | 73,12 | 76.40 |
| Infra. | Camera | 61,98 | 31.19 | 46.73 | 40.42 | 35.04 |
| Infra. | LiDAR | 92.86 | 86.17 | 88.07 | 75.73 | 84.88 |
| Infra. | Camera + LiDAR | 92.92 | 87.99 | **89.09** | **81.69** | <u>87.01</u> |
| Coop. | Camera | 68.94 | 45.41 | 42.76 | 57.83 | 45.74 |
| Coop. | LiDAR | <u>93.93</u> | <u>92.63</u> | 78.06 | 73.95 | 85.86 |
| Coop. | Camera + LiDAR | **94.22** | **93.42** | <u>88.17</u> | <u>79.94</u> | **90.76** |

### Evaluation Results of Infrastructure-only CoopDet3D vs. InfraDet3D on TUMTraf Intersection Dataset Test Set

| Model        | FOV    | Modality | mAP<sub>3D</sub> Easy | mAP<sub>3D</sub> Mod. | mAP<sub>3D</sub> Hard | mAP<sub>3D</sub> Avg. |
|--------------|-------------|----------------|--------------|--------------|--------------|--------------|
| InfraDet3D | South 1 | LiDAR | 75.81 | 47.66 | **42.16** | 55.21 |
| CoopDet3D | South 1 | LiDAR | **76.24** | **48.23** | 35.19 | **69.47** |
| InfraDet3D | South 2 | LiDAR | 38.92 | 46.60 | **43.86** | 43.13 |
| CoopDet3D | South 2 | LiDAR | **74.97** | **55.55** | 39.96 | **69.94** |
| InfraDet3D | South 1 | Camera + LiDAR | 67.08 | 31.38 | 35.17 | 44.55 |
| CoopDet3D | South 1 | Camera + LiDAR | **75.68** | **45.63** | **45.63** | **66.75** |
| InfraDet3D | South 2 | Camera + LiDAR | 58.38 | 19.73 | 33.08 | 37.06 |
| CoopDet3D | South 2 | Camera + LiDAR | **74.73** | **53.46** | **41.96** | **66.89** |

## Acknowledgement 🤝 

The codebase is built upon [BEVFusion](https://github.com/mit-han-lab/bevfusion) with vehicle-infrastructure fusion inspired by the method proposed in [PillarGrid](https://arxiv.org/pdf/2203.06319.pdf).

## Citation 📝
```
@inproceedings{zimmer2024tumtrafv2x,
  title={TUMTraf V2X Cooperative Perception Dataset},
  author={Zimmer, Walter and Wardana, Gerhard Arya and Sritharan, Suren and Zhou, Xingcheng and Song, Rui and Knoll, Alois C.},
  publisher={IEEE/CVF},
  booktitle={2024 IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## License 📜

- The CoopDet3D model is released under MIT license as found in the license file.
- The TUM Traffic Dataset (`TUMTraf`) dataset itself is released under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/). 
By downloading the dataset you agree to the [terms](https://a9-dataset.innovation-mobility.com/license) of this license.

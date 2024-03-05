""" Exclude these code lines from lidar_3d_object_detection_unsupervised.py """

import argparse
import os
from pathlib import Path


def parse_cooperative_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE",
                        default="../configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, help="By default all classes defined in the config file are used.")
    parser.add_argument("--bbox-score", type=float, default=None, help="By default the score threshold defined in the config is used -> 0.1.")
    parser.add_argument('--input_type',
                        type=str,
                        default='hard_drive',
                        help='Input type can be shm, ros or hard_drive.')
    parser.add_argument('--input_folder_path_point_clouds',
                        type=str,
                        default='input/point_clouds',
                        help=
                        'Input path to point clouds. Needs to be set if input_type = device.'
                        )
    parser.add_argument('--save_detections_openlabel',
                        action='store_true',
                        help='save detection results (in openlabel format) to <TIMESTAMP_SEC>_<TIMESTAMP_NSEC>_<SENSOR_ID>.json')
    parser.add_argument('--output_folder_path_detections',
                        type=str,
                        default="../inference/detections",
                        help='Output folder path of detections (*.json). Default: inference/detections')
    args, opts = parser.parse_known_args()
    opt = vars(args)

    return opt, opts

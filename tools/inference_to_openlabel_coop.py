import argparse
import uuid
import copy
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_coop_model

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import math
import json



next_detection_id = 0

def get_corners(yaw: float, width: float, length: float, position: np.ndarray) -> np.ndarray:
    # create the (normalized) perpendicular vectors
    v1 = np.array([np.cos(yaw), np.sin(yaw), 0])
    v2 = np.array([-v1[1], v1[0], 0])  # rotate by 90

    # scale them appropriately by the dimensions
    v1 *= length * 0.5
    v2 *= width * 0.5

    # flattened position
    pos = position.flatten()

    # return the corners by moving the center of the rectangle by the vectors
    return np.array(
        [
            pos + v1 + v2,
            pos - v1 + v2,
            pos - v1 - v2,
            pos + v1 - v2,
        ]
    )

@dataclass
class Detection:
    """Representation of a single detected road user."""

    # Detection location in [[x], [y], [z]] format.
    location: np.ndarray

    # Length, width, height
    dimensions: Tuple[float, float, float]

    # Heading angle in Radians
    yaw: float

    # Detection category (BUS, TRUCK, ...)
    category: str

    # projected 2D bottom contour (2xn)
    bottom_contour: Optional[np.ndarray] = None

    # Previous positions and heading angles known from tracking.
    yaw_history: List[float] = field(default_factory=list)
    pos_history: List[np.ndarray] = field(default_factory=list)

    # Screen-space bounding box in pixel coords
    bbox_2d: Optional[np.ndarray] = None  # [x_min, y_min, x_max, y_max]

    # Tracking ID, -1 if unknown
    id: int = -1

    # Sensor ID: ID of the sensor that detected this vehicle. Possible values:
    # - s110_lidar_ouster_south
    # - s110_lidar_ouster_north
    # - s110_camera_basler_south1_8mm,
    # - s110_camera_basler_south2_8mm
    sensor_id: str = ""

    # Unique object ID (32 hex chars)
    uuid: str = ""

    # Speed vector
    speed: Optional[np.ndarray] = None  # [dx, dy] * m/s

    # Original ROS message from which this detection was estimated
    original_detected_object_ros_msg: Optional[Any] = None

    # Detection color from 2D detection
    color: Optional[str] = None

    # Number of LiDAR points within bounding box
    num_lidar_points: int = 0

    # prediction score
    score: float = 0.0

    # Angle in s110_base (in radians)
    yaw_base: float = 0.0

    # occlusion level (NOT_OCCLUDED, PARTIALLY_OCCLUDED, MOSTLY_OCCLUDED)
    occlusion_level: str = None

    overlap: bool = False

    def __post_init__(self):
        global next_detection_id
        if self.id == -1:
            self.id = next_detection_id
        next_detection_id += 1

    def as_2d_bev_square(self, confidence=0.5):
        """Convert the detection to an AABB as expected by SORT."""
        loc = self.location.flatten()
        half_avg_size = (self.dimensions[0] * self.dimensions[1]) * 0.25
        return [
            loc[0] - half_avg_size,  # x0
            loc[1] - half_avg_size,  # y0
            loc[0] + half_avg_size,  # x1
            loc[1] + half_avg_size,  # y1
            confidence,
        ]

    def adjust_yaw(self, delta: float):
        self.yaw += delta

    def pick_yaw(self, yaw_options: np.ndarray, apply=True) -> float:
        """Align current heading with a valid option, considering
        current and past headings."""
        if not self.yaw_history:
            if apply:
                self.adjust_yaw(yaw_options[0] - self.yaw)
            return yaw_options[0]
        yaw_history = np.array(self.yaw_history)[:4]
        # Calculate difference to each option for the current and some historical yaw values.
        # The best option is determined as the one with the smallest
        # cumulative difference towards the most recent 4 historical values.
        # Difference values are in range of [-PI/2, PI/2]. The sign is maintained,
        # such that the result may be used to fix the heading later.
        half_pi = math.pi * 0.5
        delta = yaw_options.reshape((-1, 1)) - np.tile(yaw_history, len(yaw_options)).reshape((-1, len(yaw_history)))
        delta = np.mod(delta, math.pi)
        delta[delta > half_pi] -= math.pi
        delta[delta < -half_pi] += math.pi
        delta_cumulative = np.sum(np.abs(delta), axis=1)
        best_option = np.argmin(delta_cumulative)
        if apply:
            self.adjust_yaw(yaw_options[best_option] - self.yaw)
        return yaw_options[best_option]

    def get_corners(self) -> np.ndarray:
        return get_corners(self.yaw, self.dimensions[1], self.dimensions[0], self.location)

    def speed_kmh(self) -> float:
        if self.speed is None:
            return 0.0
        return np.linalg.norm(self.speed) * 3.6

    def get_bbox_2d_center(self):
        return ((self.bbox_2d[:2] + self.bbox_2d[2:]) * 0.5).reshape((2, 1))

def detections_to_openlabel(
    detection_list: List[Detection],
    filename: str,
    output_folder_path: Path,
    coordinate_systems=None,
    frame_properties=None,
    frame_id=None,
    streams=None,
):
    """
    Convert list of detected detections to the format expected by the OpenLABEL
    """
    output_json_data = {"openlabel": {"metadata": {"schema_version": "1.0.0"}, "coordinate_systems": {}}}
    if coordinate_systems:
        output_json_data["openlabel"]["coordinate_systems"] = coordinate_systems

    objects_map = {}
    if frame_id is None:
        frame_id = "0"
    frame_map = {str(frame_id): {}}
    for detection_idx, detection in enumerate(detection_list):
        category = detection.category
        position_3d = detection.location.flatten()
        rotation_yaw = detection.yaw
        rotation_quat = R.from_euler("xyz", [0, 0, rotation_yaw], degrees=False).as_quat()
        dimensions = detection.dimensions

        # TODO: store all unique track IDs (.id) into a unique list (set).
        #  Generate for each integer a unique 32-bit string that will be stored in OpenLABEL
        # TODO: Better: use existing uuid for tracking (change tracker.py from id to uuid)
        object_id = str(detection.uuid) or str(detection_idx)

        object_attributes = {"text": [], "num": [], "vec": []}

        if detection.color is not None:
            body_color_attribute = {"name": "body_color", "val": detection.color.lower()}
            object_attributes["text"].append(body_color_attribute)

        overlap_attribute = {"name": "overlap", "val": str(detection.overlap)}
        object_attributes["text"].append(overlap_attribute)

        if detection.occlusion_level is not None:
            occlusion_attribute = {"name": "occlusion_level", "val": detection.occlusion_level}
            object_attributes["text"].append(occlusion_attribute)

        if detection.sensor_id is not None:
            sensor_id_attribute = {"name": "sensor_id", "val": detection.sensor_id}
            object_attributes["text"].append(sensor_id_attribute)

        num_lidar_points_attribute = {"name": "num_points", "val": detection.num_lidar_points}
        object_attributes["num"].append(num_lidar_points_attribute)

        score_attribute = {"name": "score", "val": detection.score}
        object_attributes["num"].append(score_attribute)

        if detection.bbox_2d is not None:
            # convert x_min, y_min, x_max, y_max to xywh
            width = float(detection.bbox_2d[2] - detection.bbox_2d[0])
            height = float(detection.bbox_2d[3] - detection.bbox_2d[1])
            x_center = float(detection.bbox_2d[0] + width / 2.0)
            y_center = float(detection.bbox_2d[1] + height / 2.0)
            bbox_2d = [{"name": "shape", "val": [x_center, y_center, width, height]}]
        else:
            bbox_2d = []

        # store track history
        if detection.pos_history is not None:
            track_history = []
            for pos in detection.pos_history:
                position = pos.flatten().tolist()
                track_history.append(position[0])
                track_history.append(position[1])
                track_history.append(position[2])
            track_history_attribute = {"name": "track_history", "val": track_history}
            object_attributes["vec"].append(track_history_attribute)

        objects_map[object_id] = {
            "object_data": {
                "name": category.upper() + "_" + object_id.split("-")[0],
                "type": category.upper(),
                "cuboid": {
                    "name": "shape3D",
                    "val": [
                        position_3d[0],
                        position_3d[1],
                        position_3d[2],
                        rotation_quat[0],
                        rotation_quat[1],
                        rotation_quat[2],
                        rotation_quat[3],
                        dimensions[0],
                        dimensions[1],
                        dimensions[2],
                    ],
                    "attributes": object_attributes,
                },
                "bbox": bbox_2d,
            }
        }
    frame_map[frame_id]["objects"] = objects_map
    if frame_properties:
        frame_map[frame_id]["frame_properties"] = frame_properties
    if streams:
        output_json_data["openlabel"]["streams"] = streams
    output_json_data["openlabel"]["frames"] = frame_map
    
    with open(output_folder_path+filename, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=4)

    return output_json_data


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="openlabel_out")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_coop_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    for data in tqdm(dataflow):
        detections = []
        metas = data["metas"].data[0][0]

        pcd = o3d.geometry.PointCloud()

        vehicle_points = data["vehicle_points"].data[0][0].cpu().numpy()
        infrastructure_points = data["infrastructure_points"].data[0][0].cpu().numpy()
        o3dpoints = np.concatenate((vehicle_points, infrastructure_points), axis=0)
        pcd.points = o3d.utility.Vector3dVector(o3dpoints[:, 1:4])

        with torch.inference_mode():
            outputs = model(**data)

        bboxes = outputs[0]["boxes_3d"].tensor.numpy()
        scores = outputs[0]["scores_3d"].numpy()
        labels = outputs[0]["labels_3d"].numpy()

        if args.bbox_classes is not None:
            indices = np.isin(labels, args.bbox_classes)
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        if args.bbox_score is not None:
            indices = scores >= args.bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        for i, bbox in enumerate(bboxes):
            loc = np.asarray([[bbox[0]], [bbox[1]], [bbox[2]+(0.5*bbox[5])]]).astype(float)

            o3d_bbox = o3d.geometry.OrientedBoundingBox()
            o3d_bbox.center = bbox[:3]
            o3d_bbox.R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0, 0, bbox[6]]))
            o3d_bbox.extent = np.array([bbox[3], bbox[4], bbox[5]])
            o3d_bbox.color = np.array([1, 0, 0])
            num_lidar_points = len(o3d_bbox.get_point_indices_within_bounding_box(pcd.points))

            detections.append(
                Detection(
                    uuid=str(uuid.uuid4()),
                    category=cfg.object_classes[labels[i]],
                    location=loc,
                    dimensions=(float(bbox[3]), float(bbox[4]), float(bbox[5])),
                    yaw=-float(bbox[6]),
                    num_lidar_points=num_lidar_points,
                    score=float(scores[i]),
                    sensor_id="s110_lidar_ouster_south",
                )
            )

        fname = metas["infrastructure_lidar_path"].split("/")[-1].replace("s110_lidar_ouster_south.bin", "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json")
        detections_to_openlabel(detection_list=detections, filename=fname, output_folder_path=args.out_dir)
    

if __name__ == "__main__":
    main()

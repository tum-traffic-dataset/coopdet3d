#!/usr/bin/env python
import json


###################
# IMPORTS
###################
from copy import deepcopy
import time
import open3d
from scipy.spatial.transform import Rotation as Rotation
import cv2
import sys
import uuid
import copy
import torchvision
import os
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm
from mmcv.parallel import DataContainer as DC

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from pathlib import Path
import open3d as o3d


# inside docker
sys.path.insert(0, "/home/coopdet3d")
from mmdet3d.models import build_coop_model
from scripts.utils import id_to_class_name_mapping
from scripts.detection import Detection, detections_to_openlabel
from scripts.parse_cooperative_parameters import \
    parse_cooperative_parameters


def generate_uuids(num_uuids):
    uuids = []
    for i in range(num_uuids):
        uuids.append(str(uuid.uuid4()))
    return uuids


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


class Detector:
    def __init__(self, opt):
        self.opt = opt
        self.dataset_loaded = None
        self.input_file_path_point_cloud = None
        self.dataset = None
        self.num_min_points = None
        self.mapping_id_to_uuid = generate_uuids(10000)
        self.v2i_transformation_matrix = None

    def initialize(self):
        input_folder_path_point_clouds = self.opt["input_folder_path_point_clouds"]
        if bool(self.opt["save_detections_openlabel"]):
            subfolder = "/openlabel"
            output_folder_path_detections_obj = Path(self.opt["output_folder_path_detections"] + subfolder,
                                                     exist_ok=False)
            output_folder_path_detections_obj.mkdir(parents=True, exist_ok=True)
            # Update incremented path
            self.opt["output_folder_path_detections"] = str(output_folder_path_detections_obj)

        if self.opt["input_type"] == "hard_drive":
            input_file_path_point_cloud = input_folder_path_point_clouds
            self.input_file_path_point_cloud = input_file_path_point_cloud

    def detect(self, point_cloud, timestamp=0, frame_id=None):
        if self.opt["input_type"] == "hard_drive":
            self.pc_original = point_cloud
        else:
            print("Unknown input_type: ", self.opt["input_type"], ". Possible values are: [ros, hard_drive]")
            sys.exit()
        # Inference

        detections, boxes = self.inference(self.pc_original)

        # filter by overlap
        if bool(self.opt["filter_by_overlap"]):
            detections, boxes = self.filter_by_overlap(detections, boxes)
            # print("frame_id: ", frame_id)

            # update track IDs
            for detection in detections:
                if detection.id != -1:
                    detection.uuid = self.mapping_id_to_uuid[detection.id]

        if bool(self.opt["save_pc"]) or bool(self.opt["save_detections_kitti"]) or bool(
                self.opt["save_detections_openlabel"]) or bool(self.opt["view_pc"]):
            if self.opt["input_type"] == "hard_drive":
                file_name_point_cloud = self.input_file_path_point_cloud.split("/")[-1]
            if bool(self.opt["save_pc"]):
                output_file_path_vis_point_cloud = os.path.join(self.opt["output_folder_path_point_clouds"],
                                                                file_name_point_cloud)

            if bool(self.opt["save_detections_openlabel"]):
                output_file_path_detections = os.path.join(self.opt["output_folder_path_detections"],
                                                           file_name_point_cloud.replace(".pcd", ".json"))

        if bool(self.opt["save_detections_kitti"]):  # Write to file
            for detection in detections:
                line = tuple([detection.category] + np.hstack((detection.location, np.array(detection.dimensions),
                                                               detection.yaw)).tolist())  # Label format
                with open(output_file_path_detections, "a") as f:
                    f.write("%s " % line[0])
                    f.write(("%g " * len(line[1:])).rstrip() % line[1:] + "\n")
        if bool(self.opt["save_detections_openlabel"]):  # Write to file
            detections_to_openlabel(detections, file_name_point_cloud.replace(".jpg", ".json"),
                                    Path(output_file_path_detections).parent, frame_id=frame_id)

    def inference(self, point_cloud):
        input_dict = {
            "points": point_cloud
        }

        data_dict = self.dataset.prepare_data(data_dict=input_dict)

        with torch.no_grad():
            data_dict = self.dataset.collate_batch([data_dict])
            # load_data_to_gpu(data_dict, 0)
            start_time = time.time_ns()
            pred_dicts, _ = model.forward(data_dict)
            inference_time = time.time_ns() - start_time
            print("Inference time: ", inference_time / 1000000, "ms")
            pred_boxes = []
            pred_labels = []
            pred_scores = []
            for k, v in pred_dicts[0].items():
                if k == "pred_boxes":
                    pred_boxes = v.cpu().numpy()
                elif k == "pred_labels":
                    pred_labels = v.cpu().numpy() - 1
                elif k == "pred_scores":
                    pred_scores = v.cpu().numpy()

            pcd = open3d.geometry.PointCloud()
            points = data_dict["points"].cpu().numpy()
            pcd.points = o3d.utility.Vector3dVector(points[:, 1:4])

            detections = []
            boxes = []
            for i in range(len(pred_scores)):
                bbox = o3d.geometry.OrientedBoundingBox()
                bbox.center = pred_boxes[i][:3]
                bbox.R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0, 0, pred_boxes[i][6]]))
                bbox.extent = np.array([pred_boxes[i][4], pred_boxes[i][3], pred_boxes[i][5]])
                bbox.color = np.array([1, 0, 0])
                num_lidar_points = len(bbox.get_point_indices_within_bounding_box(pcd.points))
                score = float(pred_scores[i])
                category = str(id_to_class_name_mapping[str(pred_labels[i])]["class_label_en"]).upper()

                if num_lidar_points >= 5 and score >= cfg.MODEL.POST_PROCESSING.SCORE_THRESH:
                    # if num_lidar_points >= 5 and score >= score_threshold:
                    boxes.append(bbox)
                    location = pred_boxes[i][:3].astype(float)
                    # convert location from (3,) to (3,1)
                    location = np.expand_dims(location, axis=1)
                    detections.append(
                        Detection(
                            uuid=str(uuid.uuid4()),
                            category=category,
                            location=location,
                            dimensions=(float(pred_boxes[i][4]), float(pred_boxes[i][3]), float(pred_boxes[i][5])),
                            yaw=float(pred_boxes[i][6]),
                            num_lidar_points=num_lidar_points,
                            score=score,
                            sensor_id=self.opt["sensor_id"],
                        )
                    )
        return detections, boxes

    def save_point_cloud(self, image_cv2, output_file_path_vis_point_cloud):
        cv2.imwrite(output_file_path_vis_point_cloud, image_cv2)

    def get_3d_box(self, box_size, heading_angle, center):
        """Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (length,wide,height)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box corners
        """

        def roty(t):
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        R = roty(heading_angle)
        l, w, h = box_size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def get_corner_points(self, box: o3d.geometry.OrientedBoundingBox):
        center = np.array(box.center)
        extent = np.array(box.extent) / 2  # Half-lengths
        R = np.array(box.R)
        corners = np.empty((8, 3))
        for i in range(8):
            sign = np.array(
                [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                dtype=np.float32)
            corner = center + R @ (sign[i] * extent)
            corners[i] = corner
        return corners


def detect(data, pcd, input_type):
    if input_type == "hard_drive":
        vehicle_points = data["vehicle_points"].data[0][0].cpu().numpy()
        infrastructure_points = data["infrastructure_points"].data[0][0].cpu().numpy()
        o3dpoints = np.concatenate((vehicle_points, infrastructure_points), axis=0)
        pcd.points = o3d.utility.Vector3dVector(o3dpoints[:, 1:4])

    with torch.inference_mode():
        outputs = model(**data)
    bboxes = outputs[0]["boxes_3d"].tensor.numpy()
    scores = outputs[0]["scores_3d"].numpy()
    labels = outputs[0]["labels_3d"].numpy()
    if opt["bbox_classes"] is not None:
        indices = np.isin(labels, opt["bbox_classes"])
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
    if opt["bbox_score"] is not None:
        indices = scores >= opt["bbox_score"]
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
    return bboxes, scores, labels


def create_detection_list(bboxes, scores, labels):
    detections = []
    for i, bbox in enumerate(bboxes):
        loc = np.asarray([[bbox[0]], [bbox[1]], [bbox[2] + (0.5 * bbox[5])]]).astype(float)

        rotation_yaw_degree = bbox[6]
        rotation_yaw_radians = rotation_yaw_degree * np.pi / 180
        o3d_bbox = o3d.geometry.OrientedBoundingBox(
            bbox[:3],
            np.array(
                [
                    [np.cos(rotation_yaw_radians), -np.sin(rotation_yaw_radians), 0],
                    [np.sin(rotation_yaw_radians), np.cos(rotation_yaw_radians), 0],
                    [0, 0, 1],
                ]
            ),
            np.array([bbox[3], bbox[4], bbox[5]]),
        )
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
    return detections


if __name__ == "__main__":
    dist.init()

    opt, opts = parse_cooperative_parameters()
    configs.load(opt["config"], recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=opt["config"])

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    detector = Detector(opt)

    # build the model and load checkpoint
    model = build_coop_model(cfg.model)
    load_checkpoint(model, opt["checkpoint"], map_location="cpu")

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    if opt["input_type"] == "hard_drive":
        detector.initialize()
        # build the dataloader
        dataset = build_dataset(cfg.data[opt["split"]])
        dataflow = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=True,
            shuffle=False,
        )
        idx = 0
        for data in tqdm(dataflow):
            pcd = o3d.geometry.PointCloud()

            bboxes, scores, labels = detect(data, pcd, input_type=opt["input_type"])
            detections = create_detection_list(bboxes, scores, labels)
            metas = data["metas"].data[0][0]

            fname = metas["infrastructure_lidar_path"].split("/")[-1].replace("s110_lidar_ouster_south.bin",
                                                                              "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json")
            detections_to_openlabel(detection_list=detections, filename=fname,
                                    output_folder_path=Path(opt["output_folder_path_detections"]))

            idx += 1
    else:
        print("Unknown input_type: ", opt["input_type"], ". Possible values are: [ros, hard_drive]")
        sys.exit()

from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_neck,
    build_coop_fuser,
    build_fusion_model_headless,
    build_head
)
from mmdet3d.models import COOPFUSIONMODELS

from .base import Base3DCoopFusionModel

import matplotlib.pyplot as plt

lid_plot_idx = 0

def visualize_feature_map_lidar_fused(feature):
    global lid_plot_idx
    lid_plot_idx += 1
    feature = feature.detach().cpu().squeeze(0)
    #feature = feature[0] #For when training
    gray_scale = torch.sum(feature,0)
    gray_scale = gray_scale / feature.shape[0]
    plt.imshow(gray_scale)
    plt.savefig(str('/home/bevfusion/viz_tumtraf_featmap/features/lidar/fused/feature_map_'+str(lid_plot_idx)+'.png'), bbox_inches='tight')

@COOPFUSIONMODELS.register_module()
class BEVFusionCoop(Base3DCoopFusionModel):
    def __init__(
        self,
        vehicle: Dict[str, Any],
        infrastructure: Dict[str, Any],
        coop_fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.nodes = nn.ModuleDict()
        if vehicle.get("fusion_model") is not None:
            self.nodes["vehicle"] = nn.ModuleDict(
                {
                    "fusion_model": build_fusion_model_headless(vehicle["fusion_model"])
                }
            )
        else:
            self.nodes["vehicle"] = None
        if infrastructure.get("fusion_model") is not None:
            self.nodes["infrastructure"] = nn.ModuleDict(
                {
                    "fusion_model": build_fusion_model_headless(infrastructure["fusion_model"])
                }
            )
        else:
            self.nodes["infrastructure"] = None

        if coop_fuser is not None:
            self.coop_fuser = build_coop_fuser(coop_fuser)
        else:
            self.coop_fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        
        self.init_weights()

    @auto_fp16(apply_to=("vehicle_img", "vehicle_points", "infrastructure_img", "infrastructure_points"))
    def forward(
        self,
        vehicle_img,
        vehicle_points,
        vehicle_lidar2camera,
        vehicle_lidar2image,
        vehicle_camera_intrinsics,
        vehicle_camera2lidar,
        vehicle_img_aug_matrix,
        vehicle_lidar_aug_matrix,
        vehicle2infrastructure,
        infrastructure_img,
        infrastructure_points,
        infrastructure_lidar2camera,
        infrastructure_lidar2image,
        infrastructure_camera_intrinsics,
        infrastructure_camera2lidar,
        infrastructure_img_aug_matrix,
        infrastructure_lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(vehicle_img, list) or isinstance(infrastructure_img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                vehicle_img,
                vehicle_points,
                vehicle_lidar2camera,
                vehicle_lidar2image,
                vehicle_camera_intrinsics,
                vehicle_camera2lidar,
                vehicle_img_aug_matrix,
                vehicle_lidar_aug_matrix,
                vehicle2infrastructure,
                infrastructure_img,
                infrastructure_points,
                infrastructure_lidar2camera,
                infrastructure_lidar2image,
                infrastructure_camera_intrinsics,
                infrastructure_camera2lidar,
                infrastructure_img_aug_matrix,
                infrastructure_lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("vehicle_img", "vehicle_points", "infrastructure_img", "infrastructure_points"))
    def forward_single(
        self,
        vehicle_img,
        vehicle_points,
        vehicle_lidar2camera,
        vehicle_lidar2image,
        vehicle_camera_intrinsics,
        vehicle_camera2lidar,
        vehicle_img_aug_matrix,
        vehicle_lidar_aug_matrix,
        vehicle2infrastructure,
        infrastructure_img,
        infrastructure_points,
        infrastructure_lidar2camera,
        infrastructure_lidar2image,
        infrastructure_camera_intrinsics,
        infrastructure_camera2lidar,
        infrastructure_img_aug_matrix,
        infrastructure_lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # print shape vehicle_camera2lidar
        #print("shape of vehicle_camera2lidar in bevfusion_coop, forward_single: ", vehicle_camera2lidar.shape)
        #print("shape of infrastructure_camera2lidar in bevfusion_coop, forward_single: ", infrastructure_camera2lidar.shape)


        features = []
        batch_size = 0
        for node in (
            self.nodes if self.training else list(self.nodes.keys())[::-1]
        ):
            if node == "vehicle" and self.nodes[node] != None:
                feature, bs = self.nodes[node]["fusion_model"].forward(
                    vehicle_img,
                    vehicle_points,
                    vehicle_lidar2camera,
                    vehicle_lidar2image,
                    vehicle_camera_intrinsics,
                    vehicle_camera2lidar,
                    vehicle_img_aug_matrix,
                    vehicle_lidar_aug_matrix,
                    vehicle2infrastructure,
                    node,
                    metas
                )
            elif node == "vehicle" and self.nodes[node] == None:
                continue
            elif node == "infrastructure" and self.nodes[node] != None:
                feature, bs = self.nodes[node]["fusion_model"].forward(
                    infrastructure_img,
                    infrastructure_points,
                    infrastructure_lidar2camera,
                    infrastructure_lidar2image,
                    infrastructure_camera_intrinsics,
                    infrastructure_camera2lidar,
                    infrastructure_img_aug_matrix,
                    infrastructure_lidar_aug_matrix,
                    vehicle2infrastructure,
                    node,
                    metas
                )
            elif node == "infrastructure" and self.nodes[node] == None:
                continue
            else:
                raise ValueError(f"unsupported node: {node}")
            if self.coop_fuser is not None:
                features.append(feature)
            else:
                features.append(feature)
            batch_size = bs

        if self.coop_fuser is not None:
            x = self.coop_fuser(features)
            #if not self.training:
            #    visualize_feature_map_lidar_fused(x)
        else:
            assert len(features) == 1, features
            x = features[0]

        # TODO: if using ConfFuser, then update backbone and neck and head
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
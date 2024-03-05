from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32, BaseModule
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

import matplotlib.pyplot as plt

__all__ = ["BEVFusionHeadless"]

cam_plot_idx = 0
v_lid_plot_idx = 0
i_lid_plot_idx = 0
lid_plot_idx = 0

plot_idx = 0

def visualize_feature_lidar(points, path):
    global plot_idx
    path = path + str(plot_idx) + ".png"
    plot_idx = plot_idx + 1
    fig = plt.figure(figsize=(500,500))

    ax = plt.gca()
    ax.set_xlim((-250, 250))
    ax.set_ylim((-250, 250))
    ax.set_aspect(1)
    ax.set_axis_off()

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=15,
        c="white",
    )

    fig.savefig(
        path,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_feature_map_cam(feature):
    global cam_plot_idx
    cam_plot_idx += 1
    feature = feature.detach().cpu().squeeze(0)
    feature = feature[0] #For when training
    gray_scale = torch.sum(feature,0)
    gray_scale = gray_scale / feature.shape[0]
    plt.imshow(gray_scale)
    plt.savefig(str('/home/bevfusion/viz_tumtraf_featmap/features/camera/feature_map_cam_vehicle'+str(cam_plot_idx)+'.png'), bbox_inches='tight')

def visualize_feature_map_lidar_vehicle(feature):
    global v_lid_plot_idx
    v_lid_plot_idx += 1
    feature = feature.detach().cpu().squeeze(0)
    #feature = feature[0] #For when training
    gray_scale = torch.sum(feature,0)
    gray_scale = gray_scale / feature.shape[0]
    plt.imshow(gray_scale)
    plt.savefig(str('/home/bevfusion/viz_tumtraf_featmap/features/lidar/vehicle/feature_map_lidonly_'+str(v_lid_plot_idx)+'.png'), bbox_inches='tight')

def visualize_feature_map_lidar_infra(feature):
    global i_lid_plot_idx
    i_lid_plot_idx += 1
    feature = feature.detach().cpu().squeeze(0)
    #feature = feature[0] #For when training
    gray_scale = torch.sum(feature,0)
    gray_scale = gray_scale / feature.shape[0]
    plt.imshow(gray_scale)
    plt.savefig(str('/home/bevfusion/viz_tumtraf_featmap/features/lidar/infra/feature_map_'+str(i_lid_plot_idx)+'.png'), bbox_inches='tight')

def visualize_feature_map_lidar_fused(feature):
    global lid_plot_idx
    lid_plot_idx += 1
    feature = feature.detach().cpu().squeeze(0)
    #feature = feature[0] #For when training
    gray_scale = torch.sum(feature,0)
    gray_scale = gray_scale / feature.shape[0]
    plt.imshow(gray_scale)
    plt.savefig(str('/home/bevfusion/viz_tumtraf_featmap/features/lidar/fused/feature_map_fused_'+str(lid_plot_idx)+'.png'), bbox_inches='tight')

@FUSIONMODELS.register_module()
class BEVFusionHeadless(BaseModule):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                print("USING VOXELIZE")
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                # DynamicScatter is currently not working
                print("USING DYNAMICSCATTER")
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
        
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        
        self.train_counter = 0
        self.val_counter = 0

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        # print("BN", BN)
        # print("C", C)
        # print("H", H)
        # print("W", W)
        x = x.view(B, int(BN / B), C, H, W)
        x = self.encoders["camera"]["vtransform"](
            self.training,
            x,
            points,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            vehicle2infrastructure,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        # TODO: add param to visualize feature maps
        # print("Voxelize in")
        # visualize_feature_lidar(x[0].detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/vehicle/val_prevoxi")

        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res.to(torch.float32))
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points", "vehicle2infrastructure"))
    def forward(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        node,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # print("shape of camera2lidar in bevfusion_headless, forward: ", camera2lidar.shape)

        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs, batch_size = self.forward_single(
                img,
                points,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                vehicle2infrastructure,
                node,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs, batch_size

    @auto_fp16(apply_to=("img", "points", "vehicle2infrastructure"))
    def forward_single(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        node,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if node == "vehicle":
            vehicle2infrastructure = vehicle2infrastructure.to(torch.float32)
            batch_size = len(points)
            # Transform every point cloud in the batch to the augmented infrastructure LiDAR perspective
            for b in range(batch_size):
                points[b] = points[b].to(torch.float32)

            # TODO: add param to visualize feature maps
            # if not self.training:
            #   print("After headless v2i")
            #   visualize_feature_lidar(points[0].detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/vehicle/val_firstinv2i")

            features = []
            for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
            ):
                # TODO: add param to visualize feature maps
                # print("LiDAR in 1")
                # visualize_feature_lidar(points[0].detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/vehicle/val_lidarinloop")

                if sensor == "camera":
                    feature = self.extract_camera_features(
                        img,
                        points,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        vehicle2infrastructure,
                        metas,
                    )
                    # TODO: add param to visualize feature maps
                    # print(feature.size())
                    # visualize_feature_map_cam(feature)

                elif sensor == "lidar":
                    # TODO: add param to visualize feature maps
                    # print("LiDAR in 2")
                    # visualize_feature_lidar(points[0].detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/vehicle/val_lidarin")

                    feature = self.extract_lidar_features(points)

                    # TODO: add param to visualize feature maps
                    # print(feature.size())
                    # if not self.training:
                    #    print("LiDAR out")
                    #    visualize_feature_map_lidar_vehicle(feature)
                else:
                    raise ValueError(f"unsupported sensor: {sensor}")
                features.append(feature)

                # TODO: add param to visualize feature maps
                # print(feature.size())
                # visualize_feature_map_cam(feature)

        elif node == "infrastructure":
            vehicle2infrastructure = vehicle2infrastructure.to(torch.float32)
            batch_size = len(points)
            # Transform every point cloud in the batch to the augmented infrastructure LiDAR perspective
            for b in range(batch_size):
                points[b] = points[b].to(torch.float32)

            # TODO: add param to visualize feature maps
            # print("After headless v2i")
            # visualize_feature_lidar(points[0].detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/infra/val_firstinv2ii")

            features = []
            for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
            ):
                if sensor == "camera":
                    feature = self.extract_camera_features(
                        img,
                        points,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        vehicle2infrastructure,
                        metas,
                    )
                    # TODO: add param to visualize feature maps
                    # print(feature.size())
                    # visualize_feature_map_cam(feature)

                elif sensor == "lidar":
                    # TODO: add param to visualize feature maps
                    # print("LiDAR in")
                    # visualize_feature_lidar(points[0].detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/vehicle/val_lidarin")

                    feature = self.extract_lidar_features(points)

                    # TODO: add param to visualize feature maps
                    # print(feature.size())
                    # if not self.training:
                    #   visualize_feature_map_lidar_infra(feature)
                else:
                    raise ValueError(f"unsupported sensor: {sensor}")
                features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]
        if self.fuser is not None:
            x = self.fuser(features)

            # TODO: add param to visualize feature maps
            # if not self.training:
            #    visualize_feature_map_lidar_fused(x)

        else:
            assert len(features) == 1, features
            x = features[0]
        batch_size = x.shape[0]
        return x, batch_size

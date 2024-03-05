from glob import glob

import numpy as np

np.float = float

from pypcd import pypcd
import mmcv
import json

import os.path
import shutil
import json
from scipy.spatial.transform import Rotation


# TODO: Change name appropriately
class TUMTrafV2X2NuScenes(object):
    """TUMTraf-V2X dataset to nuScenes converter.

        This class serves as the converter to change the TUMTraf-V2X data to nuScenes format
        format.
    """

    def __init__(self,
                 splits,
                 load_dir,
                 save_dir,
                 name_format='name'):

        """
        Args:
            splits list[(str)]: Contains the different splits
            version (str): Specify the modality
            load_dir (str): Directory to load waymo raw data.
            save_dir (str): Directory to save data in Nuscenes format.
            name_format (str): Specify the output name of the converted file mmdetection3d expects names to but numbers
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.name_format = name_format
        self.label_save_dir = f'label_2'
        self.vehicle_point_cloud_save_dir = f'vehicle_lidar_robosense'
        self.infrastructure_point_cloud_save_dir = f's110_lidar_ouster_south'
        self.registered_point_cloud_save_dir = f's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered'

        self.train_set = []
        self.val_set = []
        self.test_set = []

        self.map_set_to_dir_idx = {
            'training': 0,
            'validation': 1,
            'testing': 2
        }

        self.map_version_to_dir = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }

        self.imagesets = {
            'training': self.train_set,
            'validation': self.val_set,
            'testing': self.test_set
        }

        self.occlusion_map = {
            'NOT_OCCLUDED': 0,
            'PARTIALLY_OCCLUDED': 1,
            'MOSTLY_OCCLUDED': 2
        }

        self.pickle = []

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # TODO: use just test for inference
        for split in self.splits:
            split_path = self.map_version_to_dir[split]
            self.create_folder(split)
            print(f'Converting split: {split}...')

            # Delete when testing split in dataset no longer has broken labels
            # if split == 'testing':
            #    continue

            test = False
            if split == 'testing':
                test = True

            vehicle_input_path = os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds',
                                  'vehicle_lidar_robosense', '*')
            print("vehicle_input_path: ", vehicle_input_path)
            vehicle_pcd_file_paths_sorted = sorted(
                glob(vehicle_input_path))
            print("vehicle_pcd_file_paths_sorted: ", vehicle_pcd_file_paths_sorted)
            infrastructure_pcd_file_paths_sorted = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds', 's110_lidar_ouster_south',
                             '*')))
            print("infrastructure_pcd_file_paths_sorted: ", infrastructure_pcd_file_paths_sorted)
            registered_pcd_file_paths_sorted = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds',
                             's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered', '*')))
            print("registered_pcd_file_paths_sorted: ", registered_pcd_file_paths_sorted)

            for idx, vehicle_pcd_file_path in enumerate(vehicle_pcd_file_paths_sorted):
                out_filename_no_ext = os.path.splitext(os.path.basename(vehicle_pcd_file_path))[0]
                self.save_lidar(vehicle_pcd_file_path,
                                os.path.join(self.vehicle_point_cloud_save_dir, out_filename_no_ext))
                vehicle_pcd_file_paths_sorted[idx] = os.path.join(self.vehicle_point_cloud_save_dir,
                                                                  out_filename_no_ext) + '.bin'

            for idx, infrastructure_pcd_file_path in enumerate(infrastructure_pcd_file_paths_sorted):
                out_filename_no_ext = os.path.splitext(os.path.basename(infrastructure_pcd_file_path))[0]
                self.save_lidar(infrastructure_pcd_file_path,
                                os.path.join(self.infrastructure_point_cloud_save_dir, out_filename_no_ext))
                infrastructure_pcd_file_paths_sorted[idx] = os.path.join(self.infrastructure_point_cloud_save_dir,
                                                                         out_filename_no_ext) + '.bin'

            for idz, registered_pcd_file_path in enumerate(registered_pcd_file_paths_sorted):
                out_filename_no_ext = os.path.splitext(os.path.basename(registered_pcd_file_path))[0]
                self.save_lidar(registered_pcd_file_path,
                                os.path.join(self.registered_point_cloud_save_dir, out_filename_no_ext))
                registered_pcd_file_paths_sorted[idz] = os.path.join(self.registered_point_cloud_save_dir,
                                                                     out_filename_no_ext) + '.bin'

            img_south1_list = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_south1_8mm',
                             '*')))
            img_south2_list = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_south2_8mm',
                             '*')))
            img_north_list = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_north_8mm',
                             '*')))
            img_vehicle_list = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 'vehicle_camera_basler_16mm',
                             '*')))
            # TODO: do not run this for test (because no labels). Use empty list instead
            pcd_labels_list = sorted(glob(
                os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_point_clouds',
                             's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered', '*')))

            infos_list = self._fill_infos(vehicle_pcd_file_paths_sorted, infrastructure_pcd_file_paths_sorted,
                                          registered_pcd_file_paths_sorted,
                                          img_south1_list, img_south2_list, img_north_list, img_vehicle_list,
                                          pcd_labels_list, test)

            metadata = dict(version='r1')

            if test:
                print("test sample: {}".format(len(infos_list)))
                data = dict(infos=infos_list, metadata=metadata)
                info_path = os.path.join(self.save_dir, "{}_infos_test.pkl".format('tumtraf_v2x_nusc'))
                mmcv.dump(data, info_path)
            else:
                if split == 'training':
                    print("train sample: {}".format(len(infos_list)))
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, "{}_infos_train.pkl".format('tumtraf_v2x_nusc'))
                    mmcv.dump(data, info_path)
                elif split == 'validation':
                    print("val sample: {}".format(len(infos_list)))
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, "{}_infos_val.pkl".format('tumtraf_v2x_nusc'))
                    mmcv.dump(data, info_path)

        print('\nFinished ...')

    def _fill_infos(self, vehicle_pcd_list, infrastructure_pcd_list, registered_pcd_list, img_south1_list,
                    img_south2_list, img_north_list, img_vehicle_list, pcd_labels_list, test=False):
        infos_list = []

        # INFRASTRUCTURE MATRICES
        # TODO: parse extrinsic matrices from label files
        # projection matrices
        infralidar2s1image = np.asarray(
            [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
             [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
             [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]], dtype=np.float32)

        infralidar2s2image = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                         [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                         [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)

        infralidar2n1image = np.asarray(
            [[-185.2891049687059, -1504.063395597006, -525.9215327879701, -23336.12843138125],
             [-240.2665682659353, 220.6722195428702, -1567.287260600104, 6362.243306159624],
             [0.6863989233970642, -0.4493367969989777, -0.5717979669570923, -6.750176429748535]], dtype=np.float32)

        south1intrinsics = np.asarray([[1400.3096617691212, 0.0, 967.7899705163408],
                                       [0.0, 1403.041082755918, 581.7195041357244],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)

        # transformation matrix 4x4
        south12infralidar = np.asarray([[0.41204962, -0.45377758, 0.7901276, 2.158825],
                                        [-0.9107832, -0.23010845, 0.34281868, -15.5765505],
                                        [0.02625162, -0.86089253, -0.5081085, 0.08758777],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # extrinsic matrix 3x4
        south12infralidar = south12infralidar[:-1, :]

        south2intrinsics = np.asarray([[1029.2795655594014, 0.0, 982.0311857478633],
                                       [0.0, 1122.2781391971948, 1129.1480997238505],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)

        south22infralidar = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                        [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                        [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        south22infralidar = south22infralidar[:-1, :]

        northintrinsics = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                      [0.0, 1362.7757568359375, 580.6482296623581],
                                      [0.0, 0.0, 1.0]], dtype=np.float32)

        north2infralidar = np.asarray([[-0.56460226, -0.4583457, 0.6863989, 0.64204305],
                                       [-0.8248329, 0.34314296, -0.4493365, -16.182753],
                                       [-0.02958117, -0.81986094, -0.57179797, 1.6824605],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        north2infralidar = north2infralidar[:-1, :]

        # VEHICLE MATRICES (TBD)
        # projection matrix (vehicle lidar to vehicle camera image)

        # TODO: update hard coded matrices for night inference
        vehiclelidar2image = np.asarray([[1019.929965441548, -2613.286262078907, 184.6794570200418, 370.7180273597151],
                                         [589.8963703919744, -24.09642935106967, -2623.908527352794,
                                          -139.3143336725661],
                                         [0.9841844439506531, 0.1303769648075104, 0.1199281811714172,
                                          -0.1664766669273376]], dtype=np.float32)

        vehiclecamintrinsics = np.asarray([[2726.55, 0.0, 685.235],
                                           [0.0, 2676.64, 262.745],
                                           [0.0, 0.0, 1.0]], dtype=np.float32)

        vehiclecam2lidar = np.asarray([[0.12672871, 0.12377692, 0.9841849, 0.14573078],  # TBD
                                       [-0.9912245, -0.02180046, 0.13037732, 0.19717109],
                                       [0.03759337, -0.99207014, 0.11992808, -0.02214238],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        vehiclecam2lidar = vehiclecam2lidar[:-1, :]

        for i, pcd_path in enumerate(infrastructure_pcd_list):
            # TODO: for inference, do not load labels
            json1_file = open(pcd_labels_list[i])
            json1_str = json1_file.read()
            lidar_annotation = json.loads(json1_str)

            lidar_anno_frame = {}

            for j in lidar_annotation['openlabel']['frames']:
                lidar_anno_frame = lidar_annotation['openlabel']['frames'][j]

            # TODO: use current timestamp for test (if SHM/ROS)
            # TODO: use current timestamp for point cloud file name (if SHM/ROS)
            # TODO: for inference (testing) load the current transformation matrix (v2i) from ROS (if SHM/ROS)
            info = {
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
                "location": lidar_anno_frame['frame_properties']['point_cloud_file_names'][1].split("_")[2],
                "vehicle_lidar_path": vehicle_pcd_list[i],
                "vehicle_sweeps": [],
                "infrastructure_lidar_path": pcd_path,
                "infrastructure_sweeps": [],
                "registered_lidar_path": registered_pcd_list[i],
                "registered_sweeps": [],
                "vehicle2infrastructure": lidar_anno_frame['frame_properties']['transforms'][
                    'vehicle_lidar_robosense_to_s110_lidar_ouster_south']['transform_src_to_dst']['matrix4x4'],
                "lidar_anno_path": pcd_labels_list[i],
                "vehicle_cams": dict(),
                "infrastructure_cams": dict(),
            }

            img_south1_info = {
                "data_path": img_south1_list[i],
                "type": 's110_camera_basler_south1_8mm',
                "lidar2image": infralidar2s1image,
                "sensor2lidar": south12infralidar,
                "camera_intrinsics": south1intrinsics,
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
            }

            info["infrastructure_cams"].update({'s110_camera_basler_south1_8mm': img_south1_info})

            img_south2_info = {
                "data_path": img_south2_list[i],
                "type": 's110_camera_basler_south2_8mm',
                "lidar2image": infralidar2s2image,
                "sensor2lidar": south22infralidar,
                "camera_intrinsics": south2intrinsics,
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
            }

            info["infrastructure_cams"].update({'s110_camera_basler_south2_8mm': img_south2_info})

            img_north_info = {
                "data_path": img_north_list[i],
                "type": 's110_camera_basler_north_16mm',
                "lidar2image": infralidar2n1image,
                "sensor2lidar": north2infralidar,
                "camera_intrinsics": northintrinsics,
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
            }

            info["infrastructure_cams"].update({'s110_camera_basler_north_8mm': img_north_info})

            img_vehicle_info = {
                "data_path": img_vehicle_list[i],
                "type": 'vehicle_camera_basler_16mm',
                "lidar2image": vehiclelidar2image,
                "sensor2lidar": vehiclecam2lidar,
                "camera_intrinsics": vehiclecamintrinsics,
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
            }

            info["vehicle_cams"].update({'vehicle_camera_basler_16mm': img_vehicle_info})

            # obtain annotation
            if not test:
                gt_boxes = []
                gt_names = []
                velocity = []
                valid_flag = []
                num_lidar_pts = []
                num_radar_pts = []

                for id in lidar_anno_frame['objects']:
                    object_data = lidar_anno_frame['objects'][id]['object_data']

                    loc = np.asarray(object_data['cuboid']['val'][:3], dtype=np.float32)
                    dim = np.asarray(object_data['cuboid']['val'][7:], dtype=np.float32)
                    rot = np.asarray(object_data['cuboid']['val'][3:7], dtype=np.float32)  # Quaternion in x,y,z,w

                    rot_temp = Rotation.from_quat(rot)
                    rot_temp = rot_temp.as_euler('xyz', degrees=False)

                    yaw = np.asarray(rot_temp[2], dtype=np.float32)

                    gt_box = np.concatenate([loc, dim, -yaw], axis=None)

                    gt_boxes.append(gt_box)
                    gt_names.append(object_data['type'])
                    velocity.append([0, 0])
                    valid_flag.append(True)

                    for n in object_data['cuboid']['attributes']['num']:
                        if n['name'] == 'num_points':
                            num_lidar_pts.append(n['val'])

                    num_radar_pts.append(0)

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = np.array(gt_names)
                info["gt_velocity"] = np.array(velocity).reshape(-1, 2)
                info["num_lidar_pts"] = np.array(num_lidar_pts)
                info["num_radar_pts"] = np.array(num_radar_pts)
                info["valid_flag"] = np.array(valid_flag, dtype=bool)

            infos_list.append(info)

        return infos_list

    @staticmethod
    def save_lidar(file, out_file):
        """
        Converts file from .pcd to .bin
        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
        np_ts = np.zeros((np_x.shape[0],), dtype=np.float32)
        bin_format = np.column_stack((np_x, np_y, np_z, np_i, np_ts)).flatten()
        bin_format.tofile(os.path.join(f'{out_file}.bin'))

    @staticmethod
    def save_img(file, out_file):
        """
        Copies images to new location
        Args:
            file: Path to image
            out_file: Path to new location
        """
        img_path = f'{out_file}.jpg'
        shutil.copyfile(file, img_path)

    def create_folder(self, split):
        """
        Create folder for data preprocessing.
        """
        split_path = self.map_version_to_dir[split]
        #print(split_path)
        self.infrastructure_point_cloud_save_dir = os.path.join(self.save_dir, split_path,
                                                                f'point_clouds/s110_lidar_ouster_south')
        #print(self.infrastructure_point_cloud_save_dir)
        os.makedirs(self.infrastructure_point_cloud_save_dir, exist_ok=True, mode=0o777)
        self.vehicle_point_cloud_save_dir = os.path.join(self.save_dir, split_path,
                                                         f'point_clouds/vehicle_lidar_robosense')
        #print(self.vehicle_point_cloud_save_dir)
        os.makedirs(self.vehicle_point_cloud_save_dir, exist_ok=True, mode=0o777)
        self.registered_point_cloud_save_dir = os.path.join(self.save_dir, split_path,
                                                            f'point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        #print(self.registered_point_cloud_save_dir)
        os.makedirs(self.registered_point_cloud_save_dir, exist_ok=True, mode=0o777)

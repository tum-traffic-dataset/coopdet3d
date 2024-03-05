from glob import glob
from pypcd import pypcd
import mmcv
import json
import numpy as np
import os.path
import shutil
import json
from scipy.spatial.transform import Rotation

# TODO: Change name appropriately
class TUMTraf2NuScenes(object):
    """TUMTraf dataset to nuScenes converter.

        This class serves as the converter to change the TUMTraf data to nuScenes
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
            save_dir (str): Directory to save data in nuScenes format.
            name_format (str): Specify the output name of the converted file mmdetection3d expects names to but numbers
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.name_format = name_format
        self.label_save_dir = f'label_2'
        self.point_cloud_save_dir = f'point_clouds'

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
        for split in self.splits:
            split_path = self.map_version_to_dir[split]
            self.create_folder(split)
            print(f'Converting split: {split}...')
            
            # Delete when testing split in dataset no longer has broken labels
            #if split == 'testing':
            #    continue

            test = False
            if split == 'testing':
                test = True

            #pcd_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds', 's110_lidar_ouster_south', '*')))
            pcd_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds', 's110_lidar_ouster_south_and_north_registered', '*')))
            
            for idx, pcd in enumerate(pcd_list):
                out_filename = pcd.split('/')
                out_filename = out_filename[8]
                out_filename = out_filename[:-4]
                self.save_lidar(pcd, os.path.join(self.point_cloud_save_dir, out_filename))
                pcd_list[idx] = os.path.join(self.point_cloud_save_dir, out_filename)+'.bin'
                  
            img_south1_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_south1_8mm', '*')))
            img_south2_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_south2_8mm', '*')))
            #pcd_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_point_clouds', 's110_lidar_ouster_south', '*')))
            pcd_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels', 's110_lidar_ouster_south_and_north_registered', '*')))
            #img_south1_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_images', 's110_camera_basler_south1_8mm', '*')))
            #img_south2_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_images', 's110_camera_basler_south2_8mm', '*')))
            img_south1_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels', 's110_lidar_ouster_south_and_north_registered', '*')))
            img_south2_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels', 's110_lidar_ouster_south_and_north_registered', '*')))

            infos_list = self._fill_infos(pcd_list, img_south1_list, img_south2_list, pcd_labels_list, img_south1_labels_list, img_south2_labels_list, test)

            metadata = dict(version='r1')

            if test:
                print("test sample: {}".format(len(infos_list)))
                data = dict(infos=infos_list, metadata=metadata)
                info_path = os.path.join(self.save_dir, "{}_infos_test.pkl".format('tumtraf_nusc'))
                mmcv.dump(data, info_path)
            else:
                if split == 'training':
                    print("train sample: {}".format(len(infos_list)))
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, "{}_infos_train.pkl".format('tumtraf_nusc'))
                    mmcv.dump(data, info_path)
                elif split == 'validation':
                    print("val sample: {}".format(len(infos_list)))
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, "{}_infos_val.pkl".format('tumtraf_nusc'))
                    mmcv.dump(data, info_path)

        print('\nFinished ...')

    def _fill_infos(self, pcd_list, img_south1_list, img_south2_list, pcd_labels_list, img_south1_labels_list, img_south2_labels_list, test=False):
        infos_list = []

        lidar2ego = np.asarray([[0.99011437, -0.13753536, -0.02752358, 2.3728100375737995],
                                [0.13828977, 0.99000475, 0.02768645, -16.19297517556697],
                                [0.02344061, -0.03121898, 0.99923766, -8.620000000000005],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        lidar2ego = lidar2ego[:-1, :]

        lidar2s1image = np.asarray([[7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
                                    [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
                                    [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00]], dtype=np.float32)

        lidar2s2image = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                    [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                    [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)
        
        south1intrinsics = np.asarray([[1400.3096617691212, 0.0, 967.7899705163408],
                                       [0.0, 1403.041082755918, 581.7195041357244],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        south12ego = np.asarray([[-0.06377762, -0.91003007, 0.15246652, -10.409943],
                                 [-0.41296193, -0.10492031, -0.8399004, -16.2729],
                                 [0.8820865, -0.11257353, -0.45447016, -11.557314],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south12ego = south12ego[:-1, :]

        south12lidar = np.asarray([[-0.10087585, -0.51122875, 0.88484734, 1.90816304],
                                   [-1.0776537, 0.03094424, -0.10792235, -14.05913251],
                                   [0.01956882, -0.93122171, -0.45454375, 0.72290242],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south12lidar = south12lidar[:-1, :]
        
        south2intrinsics = np.asarray([[1029.2795655594014, 0.0, 982.0311857478633],
                                       [0.0, 1122.2781391971948, 1129.1480997238505],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        south22ego = np.asarray([[0.650906, -0.7435749, 0.15303044, 4.6059465],
                                 [-0.14764456, -0.32172203, -0.935252, -15.00049],
                                 [0.74466264, 0.5861663, -0.3191956, -9.351643],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south22ego = south22ego[:-1, :]

        south22lidar = np.asarray([[0.49709212, -0.19863714, 0.64202357, -0.03734614],
                                   [-0.60406415, -0.17852863, 0.50214409, 2.52095055],
                                   [0.01173726, -0.77546627, -0.70523436, 0.54322305],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south22lidar = south22lidar[:-1, :]

        for i, pcd_path in enumerate(pcd_list):
            json1_file = open(pcd_labels_list[i])
            json1_str = json1_file.read()
            lidar_annotation = json.loads(json1_str)

            lidar_anno_frame = {}

            for j in lidar_annotation['openlabel']['frames']:
                lidar_anno_frame = lidar_annotation['openlabel']['frames'][j]

            info = {
                "lidar_path": pcd_path,
                "lidar_anno_path": pcd_labels_list[i],
                "sweeps": [],
                "cams": dict(),
                "lidar2ego": lidar2ego,
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
                "location": lidar_anno_frame['frame_properties']['point_cloud_file_names'][0].split("_")[2],
            }

            json2_file = open(img_south1_labels_list[i])
            json2_str = json2_file.read()
            south1_annotation = json.loads(json2_str)

            south1_anno_frame = {}

            for k in south1_annotation['openlabel']['frames']:
                south1_anno_frame = south1_annotation['openlabel']['frames'][k]

            img_south1_info = {
                "data_path": img_south1_list[i],
                "type": 's110_camera_basler_south1_8mm',
                "lidar2image": lidar2s1image,
                "sensor2ego": south12ego,
                "sensor2lidar": south12lidar,
                "camera_intrinsics": south1intrinsics,
                "timestamp": south1_anno_frame['frame_properties']['timestamp'],
            }
            
            info["cams"].update({'s110_camera_basler_south1_8mm': img_south1_info})

            json3_file = open(img_south2_labels_list[i])
            json3_str = json3_file.read()
            south2_annotation = json.loads(json3_str)

            south2_anno_frame = {}

            for l in south2_annotation['openlabel']['frames']:
                south2_anno_frame = south2_annotation['openlabel']['frames'][l]

            img_south2_info = {
                "data_path": img_south2_list[i],
                "type": 's110_camera_basler_south2_8mm',
                "lidar2image": lidar2s2image,
                "sensor2ego": south22ego,
                "sensor2lidar": south22lidar,
                "camera_intrinsics": south2intrinsics,
                "timestamp": south2_anno_frame['frame_properties']['timestamp'],
            }
            
            info["cams"].update({'s110_camera_basler_south2_8mm': img_south2_info})

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
                    rot = np.asarray(object_data['cuboid']['val'][3:7], dtype=np.float32) # Quaternion in x,y,z,w

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
        #dir_list1 = [f'point_clouds/s110_lidar_ouster_south']
        dir_list1 = [f'point_clouds/s110_lidar_ouster_south_and_north_registered']
        for d in dir_list1:
            self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, d)
            #print(self.point_cloud_save_dir)
            os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)

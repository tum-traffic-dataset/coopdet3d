import argparse
import os
from os import path as osp
# add package folder to pythonpath
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../..'))

from tools.data_converter.tumtraf_converter import TUMTraf2Nuscenes
from tools.data_converter.create_tumtraf_gt_database import (create_groundtruth_database)


def tumtraf_data_prep(root_path,
                      info_prefix,
                      out_dir,
                      workers,
                      splits=['training', 'validation']):
    print("split: ", splits)
    load_dir = osp.join(root_path)
    save_dir = osp.join(out_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    converter = TUMTraf2NuScenes(splits, load_dir, save_dir)
    converter.convert()
    if 'training' in splits or 'validation' in splits:
        print("creating groundtruth database")
        create_groundtruth_database("TUMTrafNuscDataset", save_dir, info_prefix,
                                    f'{save_dir}/{info_prefix}_infos_train.pkl')


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/tumtraf_v2x_cooperative_perception_dataset',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/tumtraf_v2x_cooperative_perception_dataset_processed',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--splits',
    type=str,
    default='testing',
    required=True,
    help='Specify the split to be processed. Possible values: training, validation, testing. A combination of them can be used, e.g. training,validation,testing')

args = parser.parse_args()

if __name__ == '__main__':
    tumtraf_data_prep(
        root_path=args.root_path,
        info_prefix='tumtraf_nusc',
        out_dir=args.out_dir,
        workers=args.workers,
        splits=args.splits.split(',')
    )
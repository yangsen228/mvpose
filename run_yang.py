import os
import os.path as osp
import pickle
import sys
import time
import numpy as np
import pandas as pd
import cv2
import glob

import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

from src.models.model_config import model_cfg
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from src.models.estimate3d_yang import MultiEstimator
from src.m_utils.mem_dataset import MemDataset


def export(info_dicts, model, match, img_names):
    pose3d_list = list ()
    N = len(info_dicts)
    print('num of frames: {}'.format(N))

    with torch.no_grad():
        t_start = time.time()
        for i, info in enumerate(info_dicts):
            if i % 1000 == 0:
                print('processing ({:5d}/{:5d})'.format(i,N))
            # import other info ('image_data', 'cropped_img')
            info_dict = dict()
            for cam_id in range(len(info)):
                idx = match['idx{}'.format(cam_id+1)][i]
                img_name = img_names[cam_id][idx]
                assert idx == int(os.path.basename(img_name).split('_')[1])
                img = cv2.imread(img_name)
                info[cam_id]['image_data'] = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                for person_id in range(len(info[cam_id][0])):
                    bb = np.array(info[cam_id][0][person_id]['bbox'], dtype=int)
                    cropped_img = img[bb[1]:bb[3], bb[0]:bb[2]]
                    info[cam_id][0][person_id]['cropped_img'] = cv2.cvtColor(cropped_img.copy(), cv2.COLOR_BGR2RGB)
                    info[cam_id][0][person_id]['pose2d'] = info[cam_id][0][person_id]['pose2d'][:17].reshape(-1)
                    info[cam_id][0][person_id]['heatmap_data'] = []
                    info[cam_id][0][person_id]['heatmap_bbox'] = []
                info_dict[cam_id] = info[cam_id]

            satisfied = (np.array([len(vinfo[0]) for vinfo in info]) > 0).sum()
            if satisfied > 1:
                model.dataset = MemDataset(info_dict=info_dict, camera_parameter=camera_parameter,
                                        template_name='Unified')
                poses3d = model._estimate3d(0)
            else:
                pose3d = -1
            
            pose3d_list.append(poses3d)
            info_dicts[i] = -1
        t_end = time.time()
        print('overall avg time: {}'.format((t_end - t_start) / N))
    return pose3d_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', type=str, dest='datasets', required=True)
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )

    model_cfg.testing_on = args.datasets

    # read the camera parameter of this dataset
    with open ( osp.join ( args.datasets, 'camera_parameter.pickle' ),
                'rb' ) as f:
        camera_parameter = pickle.load ( f )

    # read image file names
    views = ['cam1_low', 'cam2_mid', 'cam3_high']
    img_names = []
    for v in views:
        v_names = sorted(glob.glob(os.path.join(args.datasets, v, 'color/*.jpg')))
        img_names.append(v_names)

    # inference
    match = pd.read_csv(os.path.join(args.datasets, 'matching.csv'))
    info_dicts = np.load(os.path.join(args.datasets, 'info_dicts.npy'),allow_pickle=True)
    assert match.shape[0] == len(info_dicts)

    res = export ( info_dicts, test_model, match, img_names)
    
    with open ( osp.join ( args.datasets, 'pose3d.pkl' ), 'wb' ) as f:
        pickle.dump ( res, f )

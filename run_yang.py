import os
import os.path as osp
import pickle
import sys
import time
import numpy as np
import pandas as pd
import cv2

import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from src.models.estimate3d_yang import MultiEstimator
from src.m_utils.mem_dataset import MemDataset


def export(info_dicts, model, match):
    pose3d_list, matched_list, feat_list = list (), list(), list ()
    print('num of frames: {}'.format(len(info_dicts)))

    with torch.no_grad():
        t_start = time.time()
        for i, info in enumerate(info_dicts):
            # import other info ('image_data', 'cropped_img')
            info_dict = dict()
            for cam_id in range(len(info)):
                idx = match['idx{}'.format(cam_id+1)][i]
                # img = cv2.imread()
                # info[cam_id]['image_data'] = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                for person_id in range(len(info[cam_id][0])):
                    bb = info[cam_id][0][person_id]['bbox']
                    # cropped_img = img[bb[1]:bb[3], bb[0]:bb[2]]
                    # info[cam_id][0][person_id]['cropped_img'] = cv2.cvtColor(cropped_img.copy(), cv2.COLOR_BGR2RGB)
                info_dict[cam_id] = info[cam_id]

            model.dataset = MemDataset(info_dict=info_dict, camera_parameter=camera_parameter,
                                        template_name='Unified')
            poses3d, match, features = model._estimate3d(0)
            
            pose3d_list.append(poses3d)
            matched_list.append(match)
            feat_list.append(features)
        t_end = time.time()
        print('overall avg time: {}'.format((t_end - t_start) / len(info_dicts)))
    return pose3d_list, matched_list, feat_list


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

    # inference
    match = pd.read_csv(os.path.join(args.datasets, 'matching.csv'))
    info_dicts = np.load(os.path.join(args.datasets, 'info_dicts.npy'),allow_pickle=True)
    assert match.shape[0] == len(info_dicts)
    res = export ( info_dicts, test_model, match)
    
    output = {'pose3d':res[0], \
                'match':res[1], \
                'features':res[2]}
                # 'id':res[6], 'bbox3d':res[7]}
    with open ( osp.join ( model_cfg.root_dir, 'result',
                            time.strftime ( str ( model_cfg.testing_on ) + "_%Y_%m_%d_%H_%M",
                                            time.localtime ( time.time () ) ) + '.pkl' ), 'wb' ) as f:
        pickle.dump ( output, f )
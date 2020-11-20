import os
import os.path as osp
import pickle
import sys
sys.path.insert(0, 'backend/inter_hand/main')
sys.path.insert(0, 'backend/inter_hand')
import time

import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.evaluate import numpify
from src.m_utils.mem_dataset import MemDataset

def export(model, loader, is_info_dicts=False, show=False):
    pose3d_list, pose2d_list, hand_bbox_list, hand_pose_list = list (), list (), list(), list()
    print('num of frames: {}'.format(len(loader)))
    with torch.no_grad():
        t_start = time.time()
        for img_id, imgs in enumerate ( tqdm ( loader ) ):
            if is_info_dicts:
                info_dicts = numpify ( imgs )

                model.dataset = MemDataset ( info_dict=info_dicts, camera_parameter=camera_parameter,
                                            template_name='Unified' )
                poses3d = model._estimate3d ( 0, show=show )
            else: # √
                this_imgs = list ()  # len(this_imgs) = 3
                for img_batch in imgs:
                    this_imgs.append ( img_batch.squeeze ().numpy () )  # this_imgs[0].shape = (288, 360, 3)
                poses3d, pose2d, hand_bbox, hand_pose = model.predict ( imgs=this_imgs, camera_parameter=camera_parameter, template_name='Unified',
                                            show=show, plt_id=img_id )

            pose3d_list.append ( poses3d )
            pose2d_list.append ( pose2d )
            hand_bbox_list.append ( hand_bbox )
            hand_pose_list.append ( hand_pose )
        t_end = time.time()
        print('overall avg time: {}'.format((t_end - t_start) / len(loader)))
    return pose3d_list, pose2d_list, hand_bbox_list, hand_pose_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', nargs='+', dest='datasets', required=True,
                          choices=['Shelf', 'Campus', 'Lit', 'ultimatum1'] )
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )
    for dataset_idx, dataset_name in enumerate ( args.datasets ):
        model_cfg.testing_on = dataset_name
        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            # you can change the test_rang to visualize different images (0~3199)
            test_range = range ( 605, 1800, 5)
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            # you can change the test_rang to visualize different images (0~1999)
            test_range = [i for i in range ( 605, 1200, 6 )]
            # test_range = [i for i in range ( 1525, 1790, 2 )]
            gt_path = dataset_path

        elif dataset_name == 'Lit':
            dataset_path = model_cfg.lit_path
            # you can change the test_rang to visualize different images (0~1999)
            test_range = [i for i in range (210, 250, 1 )]  # (171, 272)
            gt_path = dataset_path 

        else:
            logger.error ( f"Unknown datasets name: {dataset_name}" )
            exit ( -1 )

        # read the camera parameter of this dataset
        with open ( osp.join ( dataset_path, 'camera_parameter.pickle' ),
                    'rb' ) as f:
            camera_parameter = pickle.load ( f )

        # using preprocessed 2D poses or using CPN to predict 2D pose
        if args.dumped_dir:
            test_dataset = PreprocessedDataset ( args.dumped_dir[dataset_idx] )
            logger.info ( f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation" )
        else:
            test_dataset = BaseDataset ( dataset_path, test_range )
            logger.info("=============================================")

        test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=6, shuffle=False )
        pose_in_range, pose_in_pixel, hand_in_bbox, hand_pose_in_range = export ( test_model, test_loader, is_info_dicts=bool ( args.dumped_dir ), show=False )
        
        output = {'pose3d':pose_in_range, 'pose2d':pose_in_pixel, 'handbox':hand_in_bbox, 'hand3d':hand_pose_in_range}
        with open ( osp.join ( model_cfg.root_dir, 'result',
                               time.strftime ( str ( model_cfg.testing_on ) + "_%Y_%m_%d_%H_%M",
                                               time.localtime ( time.time () ) ) + '.pkl' ), 'wb' ) as f:
            pickle.dump ( output, f )
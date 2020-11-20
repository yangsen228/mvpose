
import sys
import os
import os.path as osp

# Config project if not exist
project_path = osp.abspath ( osp.join ( osp.dirname ( __file__ ), '..', '..' ) )
if project_path not in sys.path:
    sys.path.insert ( 0, project_path )

from src.models.model_config import model_cfg
from backend.estimator_2d import Estimator_2d
import time
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )
import cv2
import torch
import numpy as np
from src.m_utils.geometry import geometry_affinity, get_min_reprojection_error, check_bone_length, bundle_adjustment, \
    multiTriIter
from backend.CamStyle.feature_extract import FeatureExtractor
from src.m_utils.mem_dataset import MemDataset
from src.models.matchSVT import matchSVT
from src.m_utils.visualize import show_panel_mem, plotPaperRows
# from src.models import pictorial
from src.m_lib import pictorial

# hand_bbox
from backend.yolo_hand_detection.yolo_inference import yolo_hand_detection
from backend.yolo_hand_detection.yolo import YOLO
# hand_pose
from config import cfg
from common.base import Tester
from common.utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from common.utils.transforms import flip
import torchvision.transforms as transforms

class MultiEstimator ( object ):
    def __init__(self, cfg, debug=False):
        self.est2d = Estimator_2d ( DEBUGGING=debug )
        self.extractor = FeatureExtractor ()
        self.cfg = cfg
        self.dataset = None
        # hand bbox
        self.yolo = YOLO("/home/yxs/lit/mvpose/backend/yolo_hand_detection/models/cross-hands.cfg", \
            "/home/yxs/lit/mvpose/backend/yolo_hand_detection/models/cross-hands.weights", ["hand"])
        self.sample_range = 120
        self.expand_rate = 1.5
        # hand pose
        self.tester = Tester('49')
        self.tester._make_batch_generator('test', 'all')
        self.tester._make_model()
        # coord transformation
        self.transform = lambda R, P: (R.t() @ P.t()).t()

    def expand_bbox(self, b, x_max, y_max):
        xl, xr = np.clip(int(b[0]-b[2]*(self.expand_rate-1)/2.0), 0, x_max), np.clip(int(b[0]+b[2]+b[2]*(self.expand_rate-1)/2.0), 0, x_max)
        yt, yb = np.clip(int(b[1]-b[3]*(self.expand_rate-1)/2.0), 0, y_max), np.clip(int(b[1]+b[3]+b[3]*(self.expand_rate-1)/2.0), 0, y_max)
        expanded_b = [xl,yt,xr-xl,yb-yt]
        return expanded_b

    def predict(self, imgs, camera_parameter, template_name='Shelf', show=False, plt_id=0):
        # 2d human pose estimation for all views
        info_dict = self._infer_single2d ( imgs )

        self.dataset = MemDataset ( info_dict=info_dict, camera_parameter=camera_parameter,
                                    template_name=template_name )
        
        # 3d huamn pose reconstruction
        pose3d = self._estimate3d ( 0, show=show, plt_id=plt_id )
        
        # hand bbox detection for all views
        pose2d, hand_bbox = self._detect_hand(imgs, info_dict)

        # 3d hand pose estimation
        cam_id = 1
        hand_pose = self._estimate_hand_pose(imgs, hand_bbox, camera_parameter, cam_id)
        
        return pose3d, pose2d, hand_bbox, hand_pose

    def _detect_hand(self, imgs, info_dict):
        pose2d, hand_bbox  = [], []
        for cam_id, img in enumerate(imgs):   # 遍历每个视角
            pose_tmp, hand_tmp = [], []
            for res in info_dict[cam_id][0]:  # 遍历每个人
                pose_tmp.append(res['pose2d'])
                for joint_id in [9,10]:
                    # get rough bbox
                    cx1, cy1 = res['pose2d'][joint_id*3], res['pose2d'][joint_id*3+1]  # 左/右手关节位置
                    max_x1, max_y1 = img.shape[1] - 1, img.shape[0] - 1
                    xl1, xr1 = np.clip(int(cx1-self.sample_range), 0, max_x1), np.clip(int(cx1+self.sample_range), 0, max_x1)  # rough bbox的x方向范围
                    yt1, yb1 = np.clip(int(cy1-self.sample_range), 0, max_y1), np.clip(int(cy1+self.sample_range), 0, max_y1)  # rough bbox的y方向范围
                    cropped_img = img[yt1:yb1, xl1:xr1, :]
                    # get accurate bbox
                    cnt, conf, bbox = yolo_hand_detection(self.yolo, cropped_img)
                    bbox = [[b[0]+xl1, b[1]+yt1, b[2], b[3]] for b in bbox]            # 将bbox转换到原图坐标系下 [x,y,w,h]
                    if cnt == 0:
                        hand_tmp.append(bbox)
                    else:
                        dist = [(cx1-(b[0]+b[2]/2))**2 + (cy1-(b[1]+b[3]/2))**2 for b in bbox]    # 计算每个bbox的中心到joint的距离
                        hand_tmp.append(self.expand_bbox(bbox[np.argmin(dist)], max_x1, max_y1))  # 选择距离joint最近的bbox)
            pose2d.append(pose_tmp)
            hand_bbox.append(hand_tmp)
        return pose2d, hand_bbox

    def _estimate_hand_pose(self, imgs, hand_bbox, camera_parameter, cam_id):
        img, bbox = imgs[cam_id], hand_bbox[cam_id]
        R = torch.tensor(camera_parameter['RT'][cam_id][:,:-1].astype(np.float32))  # cam_id视角的旋转矩阵
        hand_pose = []
        with torch.no_grad():
            for b in bbox:
                if len(b) > 0:
                    # crop img
                    img_patch = cv2.resize(img[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :], (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), interpolation=cv2.INTER_LINEAR)
                    img_patch = img_patch.astype(np.float32)

                    trans = transforms.ToTensor()
                    input = trans(img_patch.astype(np.float32))/255.
                    input = input.unsqueeze(0)
                    
                    inputs = {'img':input}

                    out = self.tester.model(inputs, 'test')
                    joint_coord_out = out['joint_coord'].cpu().numpy()
                    hand_type_out = out['hand_type'].cpu().numpy()

                    idx = np.argmax(hand_type_out)
                    if hand_type_out[0][idx] < 0.8:   # 忽略置信度小于0.8的结果
                        hand_pose.append([])
                        continue
                    print(hand_type_out, idx)
                    cur_pose = joint_coord_out[0, 21*idx:21*(idx+1), :] - joint_coord_out[0, 21*idx, :]  # 选择置信度更高的pose，并平移到以手腕为坐标原点的位置
                    hand_pose.append(self.transform(R, torch.tensor(cur_pose)).numpy())
                else:
                    hand_pose.append([])

        return hand_pose

    def _infer_single2d(self, imgs, img_id=0, dir='/home/jiangwen/tmp/Multi'):
        '''
        img_id一直等于０
        results[0].keys() = dict_keys(['image_id', 'category_id', 'score', 'keypoints', 'bbox', 'heatmaps', 'crops'])
        '''
        info_dict = dict ()
        for cam_id, img in enumerate ( imgs ):
            results = self.est2d.estimate_2d ( img, img_id )                                                  # type(results): list
            # results = self.est2d.light_estimate_2d(img)
            this_info_dict = {'image_data': cv2.cvtColor ( img.copy (), cv2.COLOR_BGR2RGB )}
            this_info_dict[img_id] = list ()
            for person_id, result in enumerate ( results ):                                                   # type(result): dict
                this_info_dict[img_id].append ( dict () )
                this_info_dict[img_id][person_id]['pose2d'] = result['keypoints']                             # len(result['keypoints]) = 51 -> (17*3)
                # NOTE: bbox is (x, y) (W, H) format where x and y is up-left point.
                this_info_dict[img_id][person_id]['bbox'] = result['bbox']
                bb = np.array ( result['bbox'], dtype=int )
                cropped_img = img[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
                # numpy format of crop idx is changed to json
                this_info_dict[img_id][person_id]['heatmap_bbox'] = result['crops'].astype ( int ).tolist ()  # = [x, y, x+w, y+h]
                this_info_dict[img_id][person_id]['heatmap_data'] = result['heatmaps']                        # .shape = (17, h, w)
                this_info_dict[img_id][person_id]['cropped_img'] = cv2.cvtColor ( cropped_img.copy (),
                                                                                  cv2.COLOR_BGR2RGB )

            info_dict[cam_id] = this_info_dict
        return info_dict

    def _estimate3d(self, img_id, show=False, plt_id=0):
        data_batch = self.dataset[img_id]
        # ReID affinity matrix
        affinity_mat = self.extractor.get_affinity ( data_batch, rerank=self.cfg.rerank )
        if self.cfg.rerank:
            affinity_mat = torch.from_numpy ( affinity_mat )
            affinity_mat = torch.max ( affinity_mat, affinity_mat.t () )
            affinity_mat = 1 - affinity_mat
        else:
            affinity_mat = affinity_mat.cpu ()
        dimGroup = self.dataset.dimGroup[img_id]

        info_list = list ()
        for cam_id in self.dataset.cam_names:
            info_list += self.dataset.info_dict[cam_id][img_id]    # list of dict: 每个人一个dict

        # Geometry affinity matrix
        pose_mat = np.array ( [i['pose2d'] for i in info_list] ).reshape ( -1, model_cfg.joint_num, 3 )[..., :2]   # shape = (累计人数, num_joint, 2)
        geo_affinity_mat = geometry_affinity ( pose_mat.copy (), self.dataset.F.numpy (),
                                               self.dataset.dimGroup[img_id] )
        geo_affinity_mat = torch.tensor ( geo_affinity_mat )
        if self.cfg.metric == 'geometry mean':
            W = torch.sqrt ( affinity_mat * geo_affinity_mat )
        elif self.cfg.metric == 'circle':
            W = torch.sqrt ( (affinity_mat ** 2 + geo_affinity_mat ** 2) / 2 )
        elif self.cfg.metric == 'Geometry only':
            W = torch.tensor ( geo_affinity_mat )
        elif self.cfg.metric == 'ReID only':
            W = torch.tensor ( affinity_mat )
        else:
            logger.critical ( 'Get into default option, are you intend to do it?' )
            _alpha = 0.8
            W = 0.8 * affinity_mat + (1 - _alpha) * geo_affinity_mat
        W[torch.isnan ( W )] = 0  # Some times (Shelf 452th img eg.) torch.sqrt will return nan if its too small
        sub_imgid2cam = np.zeros ( pose_mat.shape[0], dtype=np.int32 )
        for idx, i in enumerate ( range ( len ( dimGroup ) - 1 ) ):
            sub_imgid2cam[dimGroup[i]:dimGroup[i + 1]] = idx     # sub_imgid2cam[person_id] = cam_id

        num_person = 10
        X0 = torch.rand ( W.shape[0], num_person )

        # Use spectral method to initialize assignment matrix.
        if self.cfg.spectral:
            eig_value, eig_vector = W.eig ( eigenvectors=True )
            _, eig_idx = torch.sort ( eig_value[:, 0], descending=True )

            if W.shape[1] >= num_person:
                X0 = eig_vector[eig_idx[:num_person]].t ()
            else:
                X0[:, :W.shape[1]] = eig_vector.t ()



        match_mat = matchSVT ( W, dimGroup, alpha=self.cfg.alpha_SVT, _lambda=self.cfg.lambda_SVT,
                               dual_stochastic_SVT=self.cfg.dual_stochastic_SVT )


        bin_match = match_mat[:, torch.nonzero ( torch.sum ( match_mat, dim=0 ) > 1.9 ).squeeze ()] > 0.9
        bin_match = bin_match.reshape ( W.shape[0], -1 )

        matched_list = [[] for i in range ( bin_match.shape[1] )]
        for sub_imgid, row in enumerate ( bin_match ):
            if row.sum () != 0:
                pid = row.argmax ()
                matched_list[pid].append ( sub_imgid )

        matched_list = [np.array ( i ) for i in matched_list]
        if self.cfg.hybrid:
            multi_pose3d = self._hybrid_kernel ( matched_list, pose_mat, sub_imgid2cam, img_id )
            chosen_img = [[]] * len ( sub_imgid2cam )
        else:
            multi_pose3d, chosen_img = self._top_down_pose_kernel ( geo_affinity_mat, matched_list, pose_mat,
                                                                    sub_imgid2cam )
        if show:  # hybrid not implemented yet.
            bin_match = match_mat[:, torch.nonzero ( torch.sum ( match_mat, dim=0 ) > 0.9 ).squeeze ()] > 0.9
            bin_match = bin_match.reshape ( W.shape[0], -1 )
            matched_list = [[] for i in range ( bin_match.shape[1] )]
            for sub_imgid, row in enumerate ( bin_match ):
                if row.sum () != 0:
                    pid = row.argmax ()
                    matched_list[pid].append ( sub_imgid )
            matched_list = [np.array ( i ) for i in matched_list]
            show_panel_mem ( self.dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat,
                             geo_affinity_mat, W, plt_id, multi_pose3d )
            plotPaperRows ( self.dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat,
                            geo_affinity_mat, W, plt_id, multi_pose3d )

        return multi_pose3d

    def _hybrid_kernel(self, matched_list, pose_mat, sub_imgid2cam, img_id):
        return pictorial.hybrid_kernel ( self, matched_list, pose_mat, sub_imgid2cam, img_id )
        multi_pose3d = list ()

        for person in matched_list:
            # use bottom-up approach to get the 3D pose of person
            if person.shape[0] <= 1:
                continue

            # step1: use the 2D joint of person to triangulate the 3D joints candidates

            # person's 17 3D joints candidates
            candidates = np.zeros ( (17, person.shape[0] * (person.shape[0] - 1) // 2, 3) )
            # 17xC^2_nx3
            cnt = 0
            for i in range ( person.shape[0] ):
                for j in range ( i + 1, person.shape[0] ):
                    cam_id_i, cam_id_j = sub_imgid2cam[person[i]], sub_imgid2cam[person[j]]
                    projmat_i, projmat_j = self.dataset.P[cam_id_i], self.dataset.P[cam_id_j]
                    pose2d_i, pose2d_j = pose_mat[person[i]].T, pose_mat[person[j]].T
                    pose3d_homo = cv2.triangulatePoints ( projmat_i, projmat_j, pose2d_i, pose2d_j )
                    pose3d_ij = pose3d_homo[:3] / pose3d_homo[3]
                    candidates[:, cnt] += pose3d_ij.T
                    cnt += 1

            unary = self.dataset.get_unary ( person, sub_imgid2cam, candidates, img_id )

            # step2: use the max-product algorithm to inference to get the 3d joint of the person

            # change the coco order
            coco_2_skel = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            candidates = np.array ( candidates )[coco_2_skel]
            unary = unary[coco_2_skel]
            skel = pictorial.getskel ()
            # construct pictorial model
            edges = pictorial.getPictoStruct ( skel, self.dataset.distribution )
            xp = pictorial.inferPict3D_MaxProd ( unary, edges, candidates )
            human = np.array ( [candidates[i][j] for i, j in zip ( range ( xp.shape[0] ), xp )] )
            human_coco = np.zeros ( (17, 3) )
            human_coco[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = human
            human_coco[[1, 2, 3, 4]] = human_coco[0]  # Just make visualize beauty not real ear and eye
            human_coco = human_coco.T
            if self.cfg.reprojection_refine and len ( person ) > 2:
                for joint_idx in range ( human_coco.shape[1] ):
                    reprojected_error = np.zeros ( len ( person ) )
                    for idx, pid in enumerate ( person ):
                        human_coco_homo = np.ones ( 4 )
                        human_coco_homo[:3] = human_coco[:, joint_idx]
                        projected_pose_homo = self.dataset.P[sub_imgid2cam[pid]] @ human_coco_homo
                        projected_pose = projected_pose_homo[:2] / projected_pose_homo[2]
                        reprojected_error[idx] += np.linalg.norm ( projected_pose - pose_mat[pid, joint_idx] )
                    # import IPython; IPython.embed()
                    # pose_select = reprojected_error < self.cfg.refine_threshold
                    pose_select = (
                                          reprojected_error - reprojected_error.mean ()) / reprojected_error.std () < self.cfg.refine_threshold
                    if pose_select.sum () >= 2:
                        Ps = list ()
                        Ys = list ()
                        for idx, is_selected in enumerate ( pose_select ):
                            if is_selected:
                                Ps.append ( self.dataset.P[sub_imgid2cam[person[idx]]] )
                                Ys.append ( pose_mat[person[idx], joint_idx].reshape ( -1, 1 ) )
                        Ps = torch.tensor ( Ps, dtype=torch.float32 )
                        Ys = torch.tensor ( Ys, dtype=torch.float32 )
                        Xs = multiTriIter ( Ps, Ys )
                        refined_pose = (Xs[:3] / Xs[3]).numpy ()
                        human_coco[:, joint_idx] = refined_pose.reshape ( -1 )
            if True or check_bone_length ( human_coco ):
                multi_pose3d.append ( human_coco )
        return multi_pose3d

    def _top_down_pose_kernel(self, geo_affinity_mat, matched_list, pose_mat, sub_imgid2cam):
        multi_pose3d = list ()
        chosen_img = list ()
        for person in matched_list:
            Graph = geo_affinity_mat[person][:, person].clone ().numpy ()
            Graph *= (1 - np.eye ( Graph.shape[0] ))  # make diagonal 0
            if len ( Graph ) < 2:
                continue
            elif len ( Graph ) > 2:
                if self.cfg.use_mincut:
                    cut0, cut1 = find_mincut ( Graph.copy () )
                    cut = cut0 if len ( cut0 ) > len ( cut1 ) else cut1
                    cut = cut.astype ( int )
                    sub_imageid = person[cut]
                else:
                    sub_imageid = get_min_reprojection_error ( person, self.dataset, pose_mat, sub_imgid2cam )
            else:
                sub_imageid = person

            _, rank = torch.sort ( geo_affinity_mat[sub_imageid][:, sub_imageid].sum ( dim=0 ) )
            sub_imageid = sub_imageid[rank[:2]]
            cam_id_0, cam_id_1 = sub_imgid2cam[sub_imageid[0]], sub_imgid2cam[sub_imageid[1]]
            projmat_0, projmat_1 = self.dataset.P[cam_id_0], self.dataset.P[cam_id_1]
            pose2d_0, pose2d_1 = pose_mat[sub_imageid[0]].T, pose_mat[sub_imageid[1]].T
            pose3d_homo = cv2.triangulatePoints ( projmat_0, projmat_1, pose2d_0, pose2d_1 )
            if self.cfg.use_bundle:
                pose3d_homo = bundle_adjustment ( pose3d_homo, person, self.dataset, pose_mat, sub_imgid2cam,
                                                  logging=logger )
            pose3d = pose3d_homo[:3] / (pose3d_homo[3] + 10e-6)
            # pose3d -= ((pose3d[:, 11] + pose3d[:, 12]) / 2).reshape ( 3, -1 ) # No need to normalize to hip
            if check_bone_length ( pose3d ):
                multi_pose3d.append ( pose3d )
            else:
                # logging.info ( f'A pose proposal deleted on {img_id}:{person}' )
                sub_imageid = list ()
                pass
            chosen_img.append ( sub_imageid )
        return multi_pose3d, chosen_img


if __name__ == '__main__':
    import pickle
    import scipy.io as scio
    from src.models.model_config import model_cfg
    from glob import glob
    from tqdm import tqdm
    from src.m_utils.base_dataset import BaseDataset
    from torch.utils.data import DataLoader, Subset
    import random

    est = MultiEstimator ( model_cfg, debug=False )
    with open ( osp.join ( model_cfg.shelf_path, 'camera_parameter.pickle' ), 'rb' ) as f:
        test_camera_parameter = pickle.load ( f )
    test_dataset = BaseDataset ( model_cfg.shelf_path, range ( 300, 600 ) )
    test_dataset = Subset ( test_dataset, random.sample ( range ( 300 ), 50 ) )
    test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=12, shuffle=False )
    for imgs in tqdm ( test_loader ):
        this_imgs = list ()
        for img_batch in imgs:
            this_imgs.append ( img_batch.squeeze ().numpy () )
        poses3d = est.predict ( imgs=this_imgs, camera_parameter=test_camera_parameter, show=False,
                                template_name='Shelf' )
        # print ( poses3d )

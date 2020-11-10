"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
"""
import sys
import os.path as osp

project_path = osp.abspath ( osp.join ( osp.dirname ( __file__ ), '..' ) )
if project_path not in sys.path:
    sys.path.insert ( 0, project_path )

from backend.light_head_rcnn.person_detector import PersonDetector
from backend.tf_cpn.Detector2D import Detector2D
# from backend.light_openpose.pose_detector import LightDetector2D

class Estimator_2d ( object ):

    def __init__(self, DEBUGGING=False):
        self.bbox_detector = PersonDetector ( show_image=DEBUGGING )
        self.pose_detector_2d = Detector2D ( show_image=DEBUGGING )
        # self.light_detector_2d = LightDetector2D() 

    def estimate_2d(self, img, img_id):
        '''
        len(bbox_result): num of deteced objects
        bbox_result[0].keys() = dict_keys(['image_id', 'category_id', 'score', 'bbox', 'data'])
        bbox_result[0]['data'].shape = (288, 360, 3)

        len(dump_results): num of detected people
        dump_results[0].keys() = dict_keys(['image_id', 'category_id', 'score', 'keypoints', 'bbox', 'heatmaps', 'crops'])
        '''
        # import time
        # t_start = time.time()
        bbox_result = self.bbox_detector.detect ( img, img_id )
        # t_end = time.time()
        # print('bbox time: {}'.format(t_end - t_start))

        dump_results = self.pose_detector_2d.detect ( bbox_result )
        # t_end2 = time.time()
        # print('2d time: {}'.format(t_end2 - t_end))
        return dump_results

    # def light_estimate_2d(self, img):
    #     dump_results = self.light_detector_2d.detect(img)
    #     return dump_results


if __name__ == '__main__':
    import cv2

    img = cv2.imread ( 'datasets/Shelf/Camera0/img_000000.png' )
    est = Estimator_2d ()
    est.estimate_2d ( img, 0 )

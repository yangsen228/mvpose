import os
import sys
sys.path.append('./backend/yolo_hand_detection')
import cv2
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

from backend.yolo_hand_detection.yolo_inference import yolo_hand_detection
from backend.yolo_hand_detection.yolo import YOLO
yolo = YOLO("/home/yxs/lit/mvpose/backend/yolo_hand_detection/models/cross-hands.cfg", "/home/yxs/lit/mvpose/backend/yolo_hand_detection/models/cross-hands.weights", ["hand"])


skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
hand_skeleton = [[1,2],[2,3],[3,4],[4,0],[5,6],[6,7],[7,8],[8,0],[9,10],[10,11],[11,12],[12,0],[13,14],[14,15],[15,16],[16,0],[17,18],[18,19],[19,20],[20,0]]

# tmp = ['Lit_2020_10_21_11_44.pkl', 'Lit_2020_10_21_11_53.pkl', 'Lit_2020_10_21_12_00.pkl', 'Lit_2020_10_21_12_05.pkl', 'Lit_2020_10_21_12_13.pkl', \
#        'Lit_2020_10_21_12_19.pkl']

tmp = ['Lit_2020_10_21_20_58.pkl']

data = []
for f in tmp:
    path = 'result/{}'.format(f)
    dataTmp = np.load(path, allow_pickle=True)
    for i in range(len(dataTmp)):
        data.append(dataTmp[i])

data = np.array(data)
print(data.shape)

path2d = 'result/Lit2d_2020_10_21_20_58.pkl'
data2d = np.load(path2d, allow_pickle=True)  # shape = [50, 3, num_people, 51]
pathHand = '/home/yxs/lit/mvpose/backend/InterHand2.6M/output/result/mvpose_test_result.pkl'
dataHand = np.load(pathHand, allow_pickle=True)

# load images
indices = np.arange( 200, 400, 4 )
img_path = []
img_path_0 = ['datasets/LitSeq1/Camera0/LitSeq1_cam0_{:05d}.png'.format(i) for i in indices]
img_path_1 = ['datasets/LitSeq1/Camera1/LitSeq1_cam1_{:05d}.png'.format(i) for i in indices]
img_path_2 = ['datasets/LitSeq1/Camera2/LitSeq1_cam2_{:05d}.png'.format(i) for i in indices]
img_path.append(img_path_0)
img_path.append(img_path_1)
img_path.append(img_path_2)

# color setting
joint_color = (0,0,255)
bbox_color = (255,0,0)

# size setting
bbox_size = 120

# 坐标变换（转到外参标定用的棋盘格坐标系）
cam_params = np.load('/home/yxs/lit/mvpose/datasets/LitSeq1/camera_parameter.pickle', allow_pickle=True)
R = torch.tensor(cam_params['RT'][1][:,:-1].astype(np.float32))
def coord_transform(data, ref_joint):
    data = torch.tensor(data.astype(np.float32))
    transform = lambda R, P: (R @ P.t()).t()
    data = transform(R, data)
    data = data.numpy()
    data[:,:] = data[:,:] / 300.0
    data[:,:] -= data[0,:]
    data[:,:] += ref_joint
    return data

hand_crops = []
lines = []
def viz_3d_mv(preds, preds_2d, dataHand, img_path, fps=10):
    '''
    preds: [seq_len, num_views, 3, num_joints]
    out_path: video path to save
    '''
    seq_len = len(preds)

    # Draw
    fig = plt.figure()
    ax1 = fig.add_subplot(221,projection='3d')
    ax1.set_xlim3d([0,2])
    ax1.set_ylim3d([0,4])
    ax1.set_zlim3d([0,2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('skeleton')
    # ax1.elev = 180   # 改变俯仰角
    # ax1.azim = 45  # 改变相对z轴转角

    ax2 = fig.add_subplot(222)
    ax2.set_axis_off()
    ax2.set_title('camera0')

    ax3 = fig.add_subplot(223)
    ax3.set_axis_off()
    ax3.set_title('camera1')

    ax4 = fig.add_subplot(224)
    ax4.set_axis_off()
    ax4.set_title('camera2')

    sc = ax1.scatter([],[],[])
    text = ax1.text(0, -5, 0.3, "", fontsize=20, color='red')

    im0 = ax2.imshow(cv2.imread(img_path[0][0])[:,:,[2,1,0]])
    im1 = ax3.imshow(cv2.imread(img_path[1][0])[:,:,[2,1,0]])
    im2 = ax4.imshow(cv2.imread(img_path[2][0])[:,:,[2,1,0]])

    def updata(i):
        global lines
        curr_hands = []

        # reinitialization
        for l in lines:
            l[0].remove()
            del l[0]
        lines[:] = []

        # update index
        text.set_text(str(i))
        # update lines
        num_people = len(preds[i])
        sc_x, sc_y, sc_z = [], [], []
        for vi in range(num_people):
            # human body lines
            x = list(preds[i][vi][0][:])
            y = list(preds[i][vi][1][:])
            z = list(preds[i][vi][2][:])
            sc_x += x
            sc_y += y
            sc_z += z
            for idx, (child, parent) in enumerate(skeleton):
                lines.append(ax1.plot([x[child],x[parent]],[y[child],y[parent]],[z[child],z[parent]], 'r'))
            '''
            # hand lines
            right_hand, left_hand = dataHand[i][vi][0][0], dataHand[i][vi][1][0]
            if right_hand != -1:
                right_scores = right_hand['hand_type']
                if np.max(right_scores) > 0.5:
                    j = np.argmax(right_scores)
                    hand_coords = right_hand['joint_coord'][0][21*j:21*(j+1),:]
                    hand_coords = coord_transform(hand_coords, [x[9],y[9],z[9]])   # shape = (21, 3)
                    handX, handY, handZ = hand_coords[:,0], hand_coords[:,1], hand_coords[:,2]
                    # sc_x += list(handX)
                    # sc_y += list(handY)
                    # sc_z += list(handZ)
                    for idx, (child, parent) in enumerate(hand_skeleton):
                        lines.append(ax1.plot([handX[child],handX[parent]],[handY[child],handY[parent]],[handZ[child],handZ[parent]], 'r'))
            if left_hand != -1:
                left_scores = left_hand['hand_type']
                if np.max(left_scores) > 0.5:
                    j = np.argmax(left_scores)
                    hand_coords = left_hand['joint_coord'][0][21*j:21*(j+1),:]
                    hand_coords = coord_transform(hand_coords, [x[10],y[10],z[10]])   # shape = (21, 3)
                    handX, handY, handZ = hand_coords[:,0], hand_coords[:,1], hand_coords[:,2]
                    # sc_x += list(handX)
                    # sc_y += list(handY)
                    # sc_z += list(handZ)
                    for idx, (child, parent) in enumerate(hand_skeleton):
                        lines.append(ax1.plot([handX[child],handX[parent]],[handY[child],handY[parent]],[handZ[child],handZ[parent]], 'r'))
            '''
        # update joints
        sc._offsets3d = (sc_x, sc_y, sc_z)
        # update images
        img0 = cv2.imread(img_path[0][i])  # shape = (1080, 1920, 3)
        img1 = cv2.imread(img_path[1][i])
        origin_img1 = copy.deepcopy(img1)
        img2 = cv2.imread(img_path[2][i])
        # draw joints and hand_bbox
        if num_people > 0:
            pts0 = preds_2d[i][0][:][:]
            pts1 = preds_2d[i][1][:][:]
            pts2 = preds_2d[i][2][:][:]
            # for pts in pts0: # 遍历camera0中的每个人
            #     pts = np.array(pts).reshape(-1, 3)
            #     for idx, center in enumerate(pts):
            #         cx, cy = int(center[0]), int(center[1])
            #         cv2.circle(img0, (cx, cy), 10, joint_color, -1)
            #         if idx in [9, 10]:
            #             cv2.rectangle(img0, (cx-bbox_size,cy-bbox_size), (cx+bbox_size,cy+bbox_size), bbox_color, 2)
            for people_idx, pts in enumerate(pts1):
                pts = np.array(pts).reshape(-1, 3)
                for joint_idx, center in enumerate(pts):
                    cx, cy = int(center[0]), int(center[1])
                    cv2.circle(img1, (cx, cy), 10, joint_color, -1)
                    if joint_idx in [9, 10]:
                        yt, yb = np.clip(int(cy-bbox_size), 0, origin_img1.shape[0]-1), np.clip(int(cy+bbox_size), 0, origin_img1.shape[0]-1)
                        xl, xr = np.clip(int(cx-bbox_size), 0, origin_img1.shape[1]-1), np.clip(int(cx+bbox_size), 0, origin_img1.shape[1]-1)
                        # remaining task: 直接crop会有左右手图像重复的情况，最好对整张图
                        pre_cropped_img = origin_img1[yt:yb, xl:xr, :]
                        ret, conf, bbox = yolo_hand_detection(yolo, pre_cropped_img)
                        if ret > 0 and conf[0] > 0.5:
                            x, y, w, h = bbox[0]
                            cxx, cyy = x+w/2, y+h/2
                            f = 1.5
                            ytt, ybb = np.clip(int(cyy-f*h/2), 0, 2*bbox_size-1), np.clip(int(cyy+f*h/2), 0, 2*bbox_size-1)
                            xll, xrr = np.clip(int(cxx-f*w/2), 0, 2*bbox_size-1), np.clip(int(cxx+f*w/2), 0, 2*bbox_size-1)
                            cropped_img = pre_cropped_img[ytt:ybb, xll:xrr]
                            curr_hands.append(cropped_img)
                            cv2.rectangle(img1, (xl+xll,yt+ytt), (xl+xrr,yt+ybb), bbox_color, 2)
                            cv2.imwrite('./backend/InterHand2.6M/data/InterHand2.6M/images/mvpose_test/{}_{}_{}.jpg'.format(i,people_idx,joint_idx), \
                                    cropped_img)
            # for pts in pts2:
            #     pts = np.array(pts).reshape(-1, 3)
            #     for idx, center in enumerate(pts):
            #         cx, cy = int(center[0]), int(center[1])
            #         cv2.circle(img2, (cx, cy), 10, joint_color, -1)
            #         if idx in [9, 10]:
            #             cv2.rectangle(img2, (cx-bbox_size,cy-bbox_size), (cx+bbox_size,cy+bbox_size), bbox_color, 2)

        im0.set_array(img0[:,:,[2,1,0]])
        im1.set_array(img1[:,:,[2,1,0]])
        im2.set_array(img2[:,:,[2,1,0]])

    skeleton_ani = FuncAnimation(fig, updata, frames=seq_len, interval=1000/fps, repeat=True)

    plt.show()

viz_3d_mv(data, data2d, dataHand, img_path, 20)
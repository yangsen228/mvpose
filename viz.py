import os
import sys
sys.path.append('./backend/yolo_hand_detection')
import cv2
import copy
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
hand_skeleton = [[1,2],[2,3],[3,4],[4,0],[5,6],[6,7],[7,8],[8,0],[9,10],[10,11],[11,12],[12,0],[13,14],[14,15],[15,16],[16,0],[17,18],[18,19],[19,20],[20,0]]

path_list = ['Lit_2020_11_19_10_30.pkl', 'Lit_2020_11_19_16_03.pkl']

data_pose3d, data_pose2d, data_handbox, data_hand3d = [], [], [], []
for path in path_list:
    data = np.load(os.path.join('result', path), allow_pickle=True)
    for frame_id in range(len(data['pose3d'])):
        data_pose3d.append(data['pose3d'][frame_id])
        data_pose2d.append(data['pose2d'][frame_id])
        data_handbox.append(data['handbox'][frame_id])
        data_hand3d.append(data['hand3d'][frame_id])

# load images
indices = np.arange( 171, 250, 1 )
img_path = []
img_path_0 = ['datasets/LitSeq1/Camera0/LitSeq1_cam0_{:05d}.png'.format(i) for i in indices]
img_path_1 = ['datasets/LitSeq1/Camera1/LitSeq1_cam1_{:05d}.png'.format(i) for i in indices]
img_path_2 = ['datasets/LitSeq1/Camera2/LitSeq1_cam2_{:05d}.png'.format(i) for i in indices]
img_path.append(img_path_0)
img_path.append(img_path_1)
img_path.append(img_path_2)

# coordiates transform
with open ( os.path.join ( '/home/yxs/lit/mvpose/datasets/LitSeq1', 'camera_parameter.pickle' ),
                    'rb' ) as f:
    camera_parameter = pickle.load ( f )
R, T = torch.tensor(camera_parameter['RT'][1][:,:-1], dtype=torch.float32), torch.tensor(camera_parameter['RT'][1][:,-1], dtype=torch.float32)
K = torch.tensor(camera_parameter['K'][1], dtype=torch.float32)
transform = lambda R, T, K, P: (K @ (R @ P.t() + T.reshape(3,1))).t()

# color setting
joint_color = (0,0,255)
rep_color = (255,0,0)
bbox_color = (255,0,0)

hand_crops = []
lines = []
scale = 350.0
def viz_3d_mv(pose3d, pose2d, handbox, hand3d, img_path, fps=10):
    '''
    pose3d: [seq_len, num_people, 3, num_joints]
    pose2d: [seq_len, num_cam, num_people, 51=num_joint*3]
    handbox: [seq_len, num_cam, num_hand, 4]
    hand3d: [seq_len, num_hand, num_joint, 3]
    out_path: video path to save
    '''
    seq_len = len(pose3d)

    # Draw
    fig = plt.figure()
    ax1 = fig.add_subplot(221,projection='3d')
    ax1.set_xlim3d([0,1.5])
    ax1.set_ylim3d([-0.5,2.5])
    ax1.set_zlim3d([-0.5,1])
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

    sc1 = ax1.scatter([],[],[])
    sc2 = ax1.scatter([],[],[])
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
        
        # initialization
        sc1_x, sc1_y, sc1_z = [], [], []
        sc2_x, sc2_y, sc2_z = [], [], []

        # update index
        text.set_text(str(i))

        # update images
        img0 = cv2.imread(img_path[0][i])  # shape = (1080, 1920, 3)
        img1 = cv2.imread(img_path[1][i])
        img2 = cv2.imread(img_path[2][i])
        imgs = [img0, img1, img2]

        num_people = len(pose3d[i])    # pose3d[i].shape = (person_id, 3, num_joints)
        if num_people > 0:
            # reprojection (cam_id == 1)
            rep_pose3d = np.array(pose3d[i]).transpose((0,2,1)).reshape(-1,3)                      # rep_pose3d.shape = (person_id * num_joints, 3)
            rep_pose2d = transform(R,T,K,torch.tensor(rep_pose3d, dtype=torch.float32)).numpy()
            rep_pose2d = (rep_pose2d / rep_pose2d[:,-1].reshape(-1,1))[:,:-1].astype(np.int)       # rep_pose2d.shape = (person_id * num_joints, 2)
            rep_pose3d, rep_pose2d = rep_pose3d.reshape(num_people,-1,3), rep_pose2d.reshape(num_people,-1,2)
            
            for person_id in range(num_people):
                # body lines
                x = list(rep_pose3d[person_id,:,0])
                y = list(rep_pose3d[person_id,:,1])
                z = list(rep_pose3d[person_id,:,2])
                sc1_x += x
                sc1_y += y
                sc1_z += z
                for idx, (child, parent) in enumerate(skeleton):
                    lines.append(ax1.plot([x[child],x[parent]],[y[child],y[parent]],[z[child],z[parent]], 'r'))

                # hand lines
                rep_hand2dL, rep_hand2dR = rep_pose2d[person_id][9], rep_pose2d[person_id][10]
                cv2.circle(imgs[1], (rep_hand2dL[0], rep_hand2dL[1]), 10 ,rep_color, -1)                                            # cam_id == 1
                cv2.circle(imgs[1], (rep_hand2dR[0], rep_hand2dR[1]), 10 ,rep_color, -1)                                            # cam_id == 1
                distL = np.array([np.linalg.norm((rep_hand2dL - np.array([b[0]+b[2]/2, b[1]+b[3]/2]))) for b in handbox[i][1] if len(b) > 0])  # cam_id == 1
                distR = np.array([np.linalg.norm((rep_hand2dR - np.array([b[0]+b[2]/2, b[1]+b[3]/2]))) for b in handbox[i][1] if len(b) > 0])  # cam_id == 1
                if len(distL) > 0 and np.min(distL) < 100 and len(hand3d[i][np.argmin(distL)]) > 0:
                    hand_pose = hand3d[i][np.argmin(distL)]
                    xx = list(hand_pose[:,0] / scale + x[9])
                    yy = list(hand_pose[:,1] / scale + y[9])
                    zz = list(hand_pose[:,2] / scale + z[9])
                    sc2_x += xx
                    sc2_y += yy
                    sc2_z += zz
                    for idx, (child, parent) in enumerate(hand_skeleton):
                        lines.append(ax1.plot([xx[child],xx[parent]],[yy[child],yy[parent]],[zz[child],zz[parent]], 'r'))                   
                if len(distR) > 0 and np.min(distR) < 100 and len(hand3d[i][np.argmin(distR)]) > 0:
                    hand_pose = hand3d[i][np.argmin(distR)]
                    xx = list(hand_pose[:,0] / scale + x[10])
                    yy = list(hand_pose[:,1] / scale + y[10])
                    zz = list(hand_pose[:,2] / scale + z[10])
                    sc2_x += xx
                    sc2_y += yy
                    sc2_z += zz
                    for idx, (child, parent) in enumerate(hand_skeleton):
                        lines.append(ax1.plot([xx[child],xx[parent]],[yy[child],yy[parent]],[zz[child],zz[parent]], 'r'))  

        # update joints
        sc1._offsets3d = (sc1_x, sc1_y, sc1_z)
        # sc2._offsets3d = (sc2_x, sc2_y, sc2_z)
        # draw pose2d and handbox
        for cam_id in range(len(handbox[i])):
            for b_id, b in enumerate(handbox[i][cam_id]):
                if len(b) > 0:
                    cv2.rectangle(imgs[cam_id], (int(b[0]),int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), bbox_color, 2)
            #         if cam_id == 1:
            #             cv2.imwrite('/home/yxs/lit/mvpose/backend/inter_hand/data/InterHand2.6M/images/mvpose_test/{}_{}.jpg'.format(i,b_id), imgs[cam_id][b[1]:b[1]+b[3],b[0]:b[0]+b[2],:])
            for p in pose2d[i][cam_id]:
                if len(p) > 0:
                    p = np.array(p).reshape(-1,3)
                    for pts in p:
                        cv2.circle(imgs[cam_id], (int(pts[0]), int(pts[1])), 10 ,joint_color, -1)

        im0.set_array(img0[:,:,[2,1,0]])
        im1.set_array(img1[:,:,[2,1,0]])
        im2.set_array(img2[:,:,[2,1,0]])

    skeleton_ani = FuncAnimation(fig, updata, frames=seq_len, interval=1000/fps, repeat=True)

    plt.show()

viz_3d_mv(data_pose3d, data_pose2d, data_handbox, data_hand3d, img_path, 20)
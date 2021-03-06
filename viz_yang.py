import os
import sys
import cv2
import copy
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

folder = 'dance/1-1'
info_dicts = np.load(os.path.join(folder, 'info_dicts.npy'), allow_pickle=True)
data_pose3d = np.load(os.path.join(folder, 'pose3d.pkl'), allow_pickle=True)['pose3d']
match = pd.read_csv(os.path.join(folder, 'matching.csv'))

# load image paths
img_path = []
views = ['cam1_low', 'cam2_mid', 'cam3_high']
for i, v in enumerate(views):
    tmp = sorted(glob.glob(os.path.join(folder, v, 'color/*.jpg')))
    matched_idx = np.array(match['idx{}'.format(i+1)])
    img_path.append(tmp[matched_idx])

# color setting
joint_color = (0,0,255)
rep_color = (255,0,0)
bbox_color = (255,0,0)

lines = []
def viz_3d_mv(pose3d, pose2d, img_path, fps=10):
    '''
    pose3d: [seq_len, num_people, 3, num_joints]
    pose2d: [seq_len, num_cam, num_people, 51=num_joint*3]
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
    text = ax1.text(0, -1, -1.5, "", fontsize=20, color='red')

    im0 = ax2.imshow(cv2.imread(img_path[0][0])[:,:,[2,1,0]])
    im1 = ax3.imshow(cv2.imread(img_path[1][0])[:,:,[2,1,0]])
    im2 = ax4.imshow(cv2.imread(img_path[2][0])[:,:,[2,1,0]])

    def updata(i):
        global lines

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

        # update 3D joints
        num_people = len(pose3d[i])    # pose3d[i].shape = (person_id, 3, num_joints)
        if num_people > 0:
            for person_id in range(num_people):
                # body lines
                x = list(pose3d[i][person_id][0][:])
                y = list(pose3d[i][person_id][1][:])
                z = list(pose3d[i][person_id][2][:])
                sc1_x += x
                sc1_y += y
                sc1_z += z
                for idx, (child, parent) in enumerate(skeleton):
                    lines.append(ax1.plot([x[child],x[parent]],[y[child],y[parent]],[z[child],z[parent]], 'r'))

        sc1._offsets3d = (sc1_x, sc1_y, sc1_z)
        # sc2._offsets3d = (sc2_x, sc2_y, sc2_z)

        # update 2D joints and bbox
        for cam_id in range(3):
            for p in pose2d[i][cam_id][0]:
                if len(p['pose2d']) > 0:
                    p = np.array(p['pose2d'][:17]).reshape(-1,3)
                    for pts in p:
                        cv2.circle(imgs[cam_id], (int(pts[0]), int(pts[1])), 10 ,joint_color, -1)

        im0.set_array(img0[:,:,[2,1,0]])
        im1.set_array(img1[:,:,[2,1,0]])
        im2.set_array(img2[:,:,[2,1,0]])

    skeleton_ani = FuncAnimation(fig, updata, frames=seq_len, interval=1000/fps, repeat=True)

    plt.show()

viz_3d_mv(data_pose3d, info_dicts, img_path, 3)
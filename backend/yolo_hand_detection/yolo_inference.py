import argparse
import glob
import os

import cv2


def yolo_hand_detection(yolo, img, network='normal', size=640, confidence=0.6): # confidence=0.25
    yolo.size = int(size)
    yolo.confidence = float(confidence)

    detection_count = 0
    
    width, height, inference_time, results = yolo.inference(img)

    bbox, conf = [], []
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        detection_count += 1

        bbox.append([x,y,w,h])
        conf.append(confidence)

    return detection_count, conf, bbox

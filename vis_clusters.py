import os

import cv2
import numpy as np

link_pairs = [[0, 1], [1, 2], [2, 6], 
              [3, 6], [3, 4], [4, 5], 
              [6, 7], [7,12], [11, 12], 
              [10, 11], [7, 13], [13, 14],
              [14, 15],[7, 8],[8, 9]]

link_color = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
              (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (0, 255, 255), (0, 0, 255), (0, 0, 255),
              (0, 0, 255), (0, 255, 0), (0, 255, 0),
              (0, 255, 0), (0, 255, 255), (0, 255, 255)]

point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(255,255,255)]

def vis_pose(image, pose_keypoints, show=True):
    for idx, pair in enumerate(link_pairs):
        cv2.line(image, pose_keypoints[pair[0]], pose_keypoints[pair[1]], link_color[idx], 2)

    for idx, point in enumerate(pose_keypoints):
        if idx != 16:
            cv2.circle(image, point, 5, point_color[idx], thickness=-1)
        else:
            cv2.circle(image, point, 20, point_color[idx], thickness=-1)

    if show:
        cv2.imshow("image", image)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def partition(np_list, width, height, show=True):
    partitioned_image = cv2.hconcat(np_list[:width])
    for i in range(width, np_list.shape[0], width):
        partitioned_image = cv2.vconcat([partitioned_image, cv2.hconcat(np_list[i:i + width])])
    if show:
        cv2.imshow('image', partitioned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return partitioned_image

def vis_clusters(image_size, cluster_keypoints_list, partition_size=(5, 6), show=True):
    np_list = []
    background = np.zeros(image_size)
    for cluster_center in cluster_keypoints_list:
        scaled_cluster_keypoints = [(int(x[0] * image_size[0]), int(x[1] * image_size[1])) for x in cluster_center]
        image = vis_pose(background.copy(), scaled_cluster_keypoints, show=False)
        np_list.append(image)
    np_list = np.array(np_list)
    partitioned_image = partition(np_list, partition_size[0], partition_size[1], show)
    return partitioned_image

if __name__ == '__main__':

    cluster_center_path = 'affordance_data/centers_30.txt'
    image_size = (256, 256, 3)

    cluster_center_list = []
    with open(cluster_center_path, 'r') as f:
        cluster_center_list = list(f.readlines())

    cluster_keypoints_list = []
    for cluster_data in cluster_center_list:
        cluster_data = cluster_data.split(' ')[:-1]
        cluster_data = [float(x) for x in cluster_data]
        cluster_keypoints = []
        for i in range(0, len(cluster_data), 2):
            cluster_keypoints.append((cluster_data[i], cluster_data[i+1]))
        cluster_keypoints = cluster_keypoints[:-1]
        cluster_keypoints_list.append(cluster_keypoints)

    vis_clusters(image_size, cluster_keypoints_list, partition_size=(5, 6), show=True)
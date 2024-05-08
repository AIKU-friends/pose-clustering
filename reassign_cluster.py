import os

from scipy.spatial import procrustes
import numpy as np
from tqdm import tqdm

from utils import convert_series_to_keypoints

origin_data_path = './affordance_data/trainlist.txt'
cluster_data_path = './affordance_data/centers_30.txt'

cluster_center_list = []
with open(cluster_data_path, 'r') as f:
    cluster_center_list = list(f.readlines())

cluster_keypoints_list = []
for cluster_data in cluster_center_list:
    cluster_data = cluster_data.split(' ')[:-1]
    cluster_data = [float(x) for x in cluster_data]
    cluster_keypoints = []
    cluster_keypoints = cluster_data[:-2]
    cluster_keypoints_list.append(np.array(cluster_keypoints))

cnt = 0
data_list = []
with open(origin_data_path, 'r') as f:
    data_list = list(f.readlines())
data_list = [x.split(' ') for x in data_list]
for data in tqdm(data_list):
    image_name = data[0]
    pose_keypoints = data[1:-1]
    cluster = int(data[-1]) - 1

    pose_keypoints = [eval(x) for x in pose_keypoints]
    pose_keypoints = pose_keypoints[:-2]
    pose_keypoints = np.array(pose_keypoints)
    width = np.max(pose_keypoints[0::2]) - np.min(pose_keypoints[0::2])
    height = np.max(pose_keypoints[1::2]) - np.min(pose_keypoints[1::2])
    min_x = np.min(pose_keypoints[0::2])
    min_y = np.min(pose_keypoints[1::2])

    norm_pose_keypoints = []
    for idx, point in enumerate(pose_keypoints):
        if idx % 2 == 0:
            point = (point - min_x) / height
        else:
            point = (point - min_y) / height
        norm_pose_keypoints.append(point)

    # norm
    norm_pose_keypoints = np.expand_dims(np.array(norm_pose_keypoints), axis=0)
    dist = np.linalg.norm(norm_pose_keypoints - np.array(cluster_keypoints_list), axis=1)
    predicted_cluster = np.argmin(dist)
    # print(dist)
    # print(cluster)


    # procrustes

    disparity_list = []
    for cluster_keypoint in cluster_keypoints_list:
        _, _, disparity = procrustes(convert_series_to_keypoints(cluster_keypoint), convert_series_to_keypoints(norm_pose_keypoints[0]))
        _, _, disparity = procrustes(convert_series_to_keypoints(norm_pose_keypoints[0]), convert_series_to_keypoints(cluster_keypoint))
        disparity_list.append(disparity)
    predicted_cluster = np.argmin(np.array(disparity_list))
    # print(predicted_cluster)

    if predicted_cluster == cluster:
        cnt += 1

print(cnt / len(data_list) * 100)
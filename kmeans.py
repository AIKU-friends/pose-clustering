import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils import convert_series_to_keypoints, preprocess_keypoints_for_clustering, write_cluster_centers, write_pose_data, write_cluster_cnt, get_cluster_center_by_choice
from vis_clusters import vis_clusters

k_cluster = 30
partition_size = (7, 2)
random_state = 0
tag = '_kmeans_temp3'
target_data_path = os.path.join('data', tag)
if not os.path.exists(target_data_path):
    os.mkdir(target_data_path)

data_path = './affordance_data/trainlist.txt'
image_name_list, keypoints_list, norm_keypoints_list = preprocess_keypoints_for_clustering(data_path)
kmeans = KMeans(n_clusters=k_cluster, random_state=random_state, n_init="auto").fit(norm_keypoints_list)

cluster_center = kmeans.cluster_centers_
labels = kmeans.labels_
# cluster_center = get_cluster_center_by_choice(cluster_center, choice=[29, 26, 12, 27, 15, 7, 22, 5, 24, 28, 23, 13, 11, 20])
# kmeans.cluster_centers_ = cluster_center
# labels = kmeans.predict(np.array(norm_keypoints_list))

write_cluster_centers(os.path.join(target_data_path, f'centers_30{tag}.txt'), cluster_center)
write_pose_data(os.path.join(target_data_path, f'trainlist{tag}.txt'), image_name_list, keypoints_list, labels)
write_cluster_cnt(os.path.join(target_data_path, f'cluster_counts{tag}.txt'), labels)

keypoints_center_list = []
for center in cluster_center:
    keypoints_center_list.append(convert_series_to_keypoints(center))
cluster_center_image = vis_clusters(image_size=(256, 256, 3), cluster_keypoints_list=keypoints_center_list, partition_size=partition_size)
cv2.imwrite(os.path.join(target_data_path, f'cluster_centers{tag}.jpg'), cluster_center_image)

data_path = './affordance_data/testlist.txt'
image_name_list, keypoints_list, norm_keypoints_list = preprocess_keypoints_for_clustering(data_path)

labels = kmeans.predict(np.array(norm_keypoints_list))
write_pose_data(os.path.join(target_data_path, f'testlist{tag}.txt'), image_name_list, keypoints_list, labels)
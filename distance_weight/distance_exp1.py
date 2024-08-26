import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
def detect_edges(image):
    height, width = image.shape[:2]
    # edge_image = np.ones((height, width), dtype=np.uint8)
    edge_image = np.full((height, width), 255, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_pixel = image[y, x]
            neighbors = [
                image[y - 1, x - 1], image[y - 1, x], image[y - 1, x + 1],
                image[y, x - 1], current_pixel, image[y, x + 1],
                image[y + 1, x - 1], image[y + 1, x], image[y + 1, x + 1]
            ]
            if np.any([(neighbor != current_pixel).any() for neighbor in neighbors]):
                edge_image[y, x] = 0

    return edge_image

# 保存距离图像的文件夹路径

dist_folder = 'vaihingen256_stride/gtCoarse1046/exp/train'
save_path = 'vaihingen256_stride/gtCoarse1046/exp/fig'

# 创建文件夹（如果不存在）
os.makedirs(dist_folder, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

lst_path = 'vaihingen256_stride/gtCoarse1046/train_exp.lst'
edge_save_folder = 'vaihingen256_stride/gtCoarse1046/exp/edge'
os.makedirs(edge_save_folder, exist_ok=True)
with open(lst_path, 'r') as file:
    lines = file.readlines()
updated_lines = []
for line in lines:
    line_parts = line.strip().split(' ')
    image_path = line_parts[0]
    ann_path = line_parts[1]
    dist_image_path = os.path.join(dist_folder, os.path.splitext(os.path.basename(image_path))[0] + '.npy')
    edge_save_path = os.path.join(edge_save_folder, os.path.splitext(os.path.basename(image_path))[0] + "_edge.png")#image_path, _, ann_path = line.strip().split(' ')
    img_edge = cv2.imread(ann_path)
    img = cv2.imread(ann_path, cv2.IMREAD_UNCHANGED)
    # 打印图像数据的类型和形状
    edge_image = detect_edges(img_edge)
    cv2.imwrite(edge_save_path, edge_image)

    lane_image = np.copy(img)



    dist_transform = cv2.distanceTransform(edge_image, cv2.DIST_L2, 5)

    dist_transform = 1 - np.exp(-dist_transform)

    extent = [0, 512, 0, 512]
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)

    fig, ax1 = plt.subplots(figsize=(5, 5))  # 调整显示尺寸
    ax1.imshow(dist_transform, cmap='rainbow', extent=extent)
    ax1.axis('off')  # 隐藏坐标轴

    # 修改保存路径和文件名
    dist_gaussian_path = os.path.join(save_path, os.path.splitext(os.path.basename(image_path))[0] + '_weight.png')
    plt.savefig(dist_gaussian_path, bbox_inches='tight', pad_inches=0, transparent=True)

    plt.close()

    # 保存距离图像

    np.save(dist_image_path, dist_transform)

    # 更新train.lst中的第三列，添加边缘距离图像的路径

    updated_line = f"{image_path} {ann_path} {os.path.splitext(dist_image_path)[0] + '.npy'}\n"
    updated_lines.append(updated_line)

    with open(lst_path, 'w') as file:
        file.writelines(updated_lines)


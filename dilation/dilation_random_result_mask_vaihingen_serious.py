import os
import numpy as np
import cv2

def convert_to_mask(image, color_map):
    h, w, c = image.shape
    flatten_v = np.matmul(image.reshape(-1, c), np.array([2, 3, 4]).reshape(3, 1))
    out = np.zeros_like(flatten_v)
    for idx, class_color in enumerate(color_map):
        value_idx = np.matmul(class_color, np.array([2, 3, 4]).reshape(3, 1))
        out[flatten_v == value_idx] = idx
    mask = out.reshape(h, w)
    return mask

# input_folder = 'vaihingen256_stride/gtCoarse1046_serious/train_color'
# output_folder = 'vaihingen256_stride/gtCoarse1046_serious/train'

input_folder = 'vaihingen256_stride/gtCoarse1046/train_color'
output_folder = 'vaihingen256_stride/gtCoarse1046/train'
os.makedirs(output_folder, exist_ok=True)

# color_map = np.array([[0, 0, 255], [255, 255, 255], [255, 0, 0],
#                       [255, 255, 0], [0, 255, 0], [0, 255, 255]])
color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])

# 遍历文件夹中的图片
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 读取图片
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 将彩色图片转换为类别索引的mask
        mask = convert_to_mask(image, color_map)

        # 构建输出文件路径
        mask_filename = filename.replace("_coarse_color.png", "_coarse_ground.png")
        output_path = os.path.join(output_folder, mask_filename)

        # 保存mask图像
        cv2.imwrite(output_path, mask)
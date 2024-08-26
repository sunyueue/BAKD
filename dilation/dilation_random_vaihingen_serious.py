import numpy as np
import cv2
import random
import os
def dilate_erode_color(image, class_labels, iterations=1):
    result_dilate = np.zeros_like(image)
    # result_erode = np.zeros_like(image)

    for class_label in class_labels:
        # 将指定类别的像素设为前景
        class_mask = np.all(image == class_label, axis=-1).astype(np.uint8) * 255
        # 进行膨胀操作
        if np.array_equal(class_label, [255, 255, 255]):  # water
            num_dilations = random.randint(4, 6)
        elif np.array_equal(class_label, [0, 0, 255]):  # clutter
            num_dilations = random.randint(1, 3)
        elif np.array_equal(class_label, [0, 255, 255]):  # car
            num_dilations = random.randint(4, 6)
        elif np.array_equal(class_label, [0, 255, 0]):  # tree
            num_dilations = random.randint(7, 15)
        elif np.array_equal(class_label, [255, 0, 0]):  # building
            num_dilations = random.randint(7, 15)
        elif np.array_equal(class_label, [255, 255, 0]):  # 植物[255,255,0]
            num_dilations = random.randint(5, 15)
        else:
            num_dilations = random.randint(0, 0)  # 植物[255,255,0]
        dilated = cv2.dilate(class_mask, None, iterations=num_dilations)
        result_dilate[dilated == 255] = class_label

    return result_dilate


color_map = np.array([[255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])

# random.shuffle(color_map)

# color_map = random.sample(list(color_map), len(color_map))

# color_map = color_map[shuffled_indices]

input_folder = 'vaihingen256_stride/ann_dir_color/train'
output_folder = 'vaihingen256_stride/gtCoarse1046/train_color'
os.makedirs(output_folder, exist_ok=True)


# 定义所有类别
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 读取图片
        shuffled_indices = np.random.permutation(len(color_map))
        # 创建一个新的color_map副本
        shuffled_color_map = color_map.copy()

        # 使用random.sample随机打乱颜色顺序
        shuffled_color_map = random.sample(list(shuffled_color_map), len(shuffled_color_map))

        color_map = shuffled_color_map
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        class_labels = color_map

# 对所有类别的边界进行膨胀和侵蚀操作
        dilated_image = dilate_erode_color(image, class_labels, iterations=1)

        filename = filename.replace("_gtFine_ground.png", "_coarse_color.png")
        output_path = os.path.join(output_folder, filename)

        # 保存结果图像
        cv2.imwrite(output_path, dilated_image)

import os
import cv2

folder_path = '../datasets/xray2bottle/trainB'

for filename in os.listdir(folder_path):
    # 获取文件的完整路径
    file_path = os.path.join(folder_path, filename)
    # 判断是否为文件
    if os.path.isfile(file_path):
        # 获取文件的扩展名
        img = cv2.imread(file_path)
        new_image = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(file_path, new_image)
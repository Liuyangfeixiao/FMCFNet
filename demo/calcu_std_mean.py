import tqdm
import numpy as np
import cv2
import glob

def get_mean_std(img_path_list):
    # 打印出所有图片的数量
    print('Total images size:', len(img_path_list))
    # 结果向量的初始化,三个维度，和图像一样
    mean, std = np.zeros(3), np.zeros(3)
    for image_path in tqdm.tqdm(img_path_list):  # tqdm用于加载进度条
        # 读取TRAIN中的每一张图片
        image = cv2.imread(image_path)
        # 分别处理三通道
        for c in range(3):
            # 计算每个通道的均值和方差
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
    
    # 所有图像的均值和方差
    mean /= len(img_path_list)
    std /= len(img_path_list)

    return mean, std

def main():
    rgb_path_list = []
    inf_path_list = []
    rgb_path_list.extend(glob.glob(r'data/DroneVehicle/train/trainimg/*'))
    inf_path_list.extend(glob.glob(r'data/DroneVehicle/train/trainimgr/*'))

    mean_rgb, std_rgb = get_mean_std(rgb_path_list)
    mean_inf, std_inf = get_mean_std(inf_path_list)
    print('mean_rgb: ', mean_rgb)
    print('std_rgb: ', std_rgb)

    print('mean_inf: ', mean_inf)
    print('std_inf: ', std_inf)

if __name__ == '__main__':
    main()
import cv2
import numpy as np

def main():
    img_file = 'demo/00027_rgb.jpg'
    # 加载图片
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # 获取图像大小
    rows, cols = image.shape

    # 创建掩码，中心为1，其余为0
    # mask = np.ones((rows, cols), np.uint8)
    mask = np.zeros((rows, cols), np.uint8)
    center_rows = rows // 2
    center_cols = cols // 2
    mask[center_rows - 20:center_rows + 20, center_cols - 20:center_cols + 20] = 1
    mask = mask[..., None]
    # 对图片进行傅里叶变换
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 应用掩码和逆傅里叶变换
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化数据
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    cv2.imwrite('./demo/out_dir/rgb_LF_20.jpg', img_back)

if __name__ == '__main__':
    main()
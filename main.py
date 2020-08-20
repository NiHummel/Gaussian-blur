import cv2
import sys
import numpy as np
import time

# preprocessed weights of gaussian kernel with sigma = 1 and sigma = 2
gaussian_1D_kernel_with_sigma_1 = np.asarray([30, 131, 216, 131, 30], dtype=np.uint8)
gaussian_1D_kernel_with_sigma_2 = np.asarray([147, 213, 242, 213, 147], dtype=np.uint8)


def gaussian_filter(image, sigma=1):
    kernel = gaussian_1D_kernel_with_sigma_1 if sigma == 1 else gaussian_1D_kernel_with_sigma_2
    normalizer = 0
    for i in kernel:
        normalizer += i
    height, weight = image.shape[0], image.shape[1]

    # horizontal blur
    blurred_horizontal_image = np.ndarray(shape=(height, weight), dtype=np.uint8)
    for i in range(height):
        for j in range(weight):
            y_min, y_max = max(0, j - 2), min(weight, j + 3)
            new_pixel = 0
            for ind in range(y_min, y_max):
                new_pixel += image[i, ind] * int(kernel[ind - y_min])
            blurred_horizontal_image[i, j] = new_pixel // normalizer

    # vertical blur
    blurred_image = np.ndarray(shape=(height, weight), dtype=np.uint8)
    for i in range(height):
        x_min, x_max = max(0, i - 2), min(height, i + 3)
        for j in range(weight):
            new_pixel = 0
            for ind in range(x_min, x_max):
                new_pixel += image[ind, j] * int(kernel[ind - x_min])
            blurred_image[i, j] = new_pixel // normalizer

    return blurred_image


def main():
    assert len(sys.argv) - 1 == 1, 'Wrong amount of arguments'

    img_name = sys.argv[1]

    try:
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)
    except FileNotFoundError as e:
        raise e

    cv2.imwrite('grayscale.jpg', image)
    cv2.imwrite('blurred1.jpg', gaussian_filter(image, sigma=1))
    cv2.imwrite('blurred2.jpg', gaussian_filter(image, sigma=2))


if __name__ == '__main__':
    main()

import cv2
import sys
import numpy as np

# preprocessed weights of gaussian kernel with sigma = 1
filter14 = np.array([[30, 131, 216, 131, 30],
                     [131, 586, 966, 586, 131],
                     [216, 966, 1592, 966, 216],
                     [131, 586, 966, 586, 131],
                     [30, 131, 216, 131, 30]])

# preprocessed weights of gaussian kernel with sigma = 2
filter24 = np.array([[147, 213, 242, 213, 147],
                     [213, 310, 352, 310, 213],
                     [242, 352, 398, 352, 242],
                     [213, 310, 352, 310, 213],
                     [147, 213, 242, 213, 147]])


def main():
    assert len(sys.argv) - 1 == 1, 'Wrong amount of arguments'

    img_name = sys.argv[1]

    try:
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)
    except FileNotFoundError as e:
        raise e

    assert image.shape == (1080, 1920), 'Wrong shape of image'
    cv2.imwrite(f'grayscale.jpg', image)

    image14, image24 = [], []
    norm14, norm24 = np.sum(filter14), np.sum(filter24)

    for i in range(1080):
        row14, row24 = [], []
        xmi, xma = max(0, i - 2), min(1080, i + 3)
        for j in range(1920):
            ymi, yma = max(0, j - 2), min(1920, j + 3)
            sub_img = image[xmi:xma, ymi:yma]
            if sub_img.shape == (5, 5):
                row14.append(np.sum(np.multiply(sub_img, filter14) // norm14))
                row24.append(np.sum(np.multiply(sub_img, filter24) // norm24))
                continue
            xa, xb, ya, yb = 2 - (i - xmi), 2 + (xma - i), 2 - (j - ymi), 2 + (yma - j)
            kernel14 = filter14[xa:xb, ya:yb]
            kernel24 = filter24[xa:xb, ya:yb]
            row14.append(np.sum(np.multiply(sub_img, kernel14) // norm14))
            row24.append(np.sum(np.multiply(sub_img, kernel24) // norm24))
        image14.append(row14)
        image24.append(row24)

    cv2.imwrite('blurred1.jpg', np.asarray(image14, dtype=np.uint8))
    cv2.imwrite('blurred2.jpg', np.asarray(image24, dtype=np.uint8))


if __name__ == '__main__':
    main()

import cv2
import sys
import time
import numpy as np

# preprocessed weights of gaussian kernel with sigma = 1
filter14 = [30, 131, 216, 131, 30,
            131, 586, 966, 586, 131,
            216, 966, 1592, 966, 216,
            131, 586, 966, 586, 131,
            30, 131, 216, 131, 30]

# preprocessed weights of gaussian kernel with sigma = 2
filter24 = [147, 213, 242, 213, 147,
            213, 310, 352, 310, 213,
            242, 352, 398, 352, 242,
            213, 310, 352, 310, 213,
            147, 213, 242, 213, 147]


def main():
    assert len(sys.argv) - 1 == 1,  'Wrong amount of arguments'

    img_name = sys.argv[1]

    try:
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)
    except FileNotFoundError as e:
        raise e

    assert image.shape == (1080, 1920), 'Wrong shape of image'
    cv2.imwrite('cat2_gray_0.jpg', image)

    print(f'image processing start: {time.time()}')

    image14, image24, row14, row24 = [], [], [], []
    norm14, norm24 = sum(filter14), sum(filter24)
    buffer = []
    next_row = iter(image)
    for i in range(3):
        buffer.append(next(next_row))

    for i in range(2):
        row14, row24 = [], []
        for j in range(1920):
            t14, t24 = 0, 0
            for xy in range(10 - 5 * i, 25):
                x, y = xy // 5 - 2, xy % 5 - 2
                if 0 <= j + y < 1920:
                    cur = buffer[i + x][j + y]
                    t14 += filter14[xy] * cur
                    t24 += filter24[xy] * cur
            row14.append(t14 // norm14)
            row24.append(t24 // norm24)
        image14.append(row14)
        image24.append(row24)
        buffer.append(next(next_row))

    for i in range(2, 1077):
        row14, row24 = [], []

        t14, t24 = 0, 0
        for xy in range(0, 25):
            if 2 <= xy % 5:
                cur = buffer[xy // 5][xy % 5 - 2]
                t14 += filter14[xy] * cur
                t24 += filter24[xy] * cur
        row14.append(t14 // norm14)
        row24.append(t24 // norm24)
        t14, t24 = 0, 0
        for xy in range(0, 25):
            if 1 <= xy % 5:
                cur = buffer[xy // 5][xy % 5 - 1]
                t14 += filter14[xy] * cur
                t24 += filter24[xy] * cur
        row14.append(t14 // norm14)
        row24.append(t24 // norm24)

        for j in range(2, 1918):
            t14, t24 = 0, 0
            for xy in range(0, 25):
                cur = buffer[xy // 5][j + xy % 5 - 2]
                t14 += filter14[xy] * cur
                t24 += filter24[xy] * cur
            row14.append(t14 // norm14)
            row24.append(t24 // norm24)

        t14, t24 = 0, 0
        for xy in range(0, 25):
            if xy % 5 < 4:
                cur = buffer[xy // 5][1916 + xy % 5]
                t14 += filter14[xy] * cur
                t24 += filter24[xy] * cur
        row14.append(t14 // norm14)
        row24.append(t24 // norm24)
        t14, t24 = 0, 0
        for xy in range(0, 25):
            if xy % 5 < 3:
                cur = buffer[xy // 5][1917 + xy % 5]
                t14 += filter14[xy] * cur
                t24 += filter24[xy] * cur
        row14.append(t14 // norm14)
        row24.append(t24 // norm24)

        image14.append(row14)
        image24.append(row24)
        for r in range(4):
            buffer[r] = buffer[r + 1]
        buffer[4] = next(next_row)

    for i in range(1077, 1080):
        row14, row24 = [], []
        for j in range(1920):
            t14, t24 = 0, 0
            for xy in range(0, 25 - 5 * (i - 1077)):
                x, y = xy // 5 - 2, xy % 5 - 2
                if 0 <= j + y < 1920:
                    cur = buffer[x + 2][j + y]
                    t14 += filter14[xy] * cur
                    t24 += filter24[xy] * cur
            row14.append(t14 // norm14)
            row24.append(t24 // norm24)
        image14.append(row14)
        image24.append(row24)
        for r in range(4):
            buffer[r] = buffer[r + 1]

    print(f'image processing done: {time.time()}')

    cv2.imwrite('cat2_gray_1.jpg', np.asarray(image14, dtype=np.uint8))
    cv2.imwrite('cat2_gray_2.jpg', np.asarray(image24, dtype=np.uint8))

    print(f'image saved at: {time.time()}')


if __name__ == '__main__':
    main()

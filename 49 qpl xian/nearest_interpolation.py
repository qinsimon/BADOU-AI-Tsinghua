import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

'''
    1. use cv2--------create input image with dtype = np.uint8
    2. use matplotlib----------normlize pixel to [0,1](div 255), or plt.imshow(.astype(uint8))
    3. running time: almost 1s
'''


def nearest_interpolation(image_path, height, weight):
    original_image = cv2.imread(image_path)
    # original_image1 = plt.imread(image_path)
    H, W, _ = original_image.shape
    # H1, W1, _ = original_image.shape
    dst_h, dst_w = height, weight
    new_image = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)# dtype = np.uint8 is necessary!!!
    new_image1 = np.zeros((dst_h, dst_w, 3))
    ratio_h = H / dst_h
    ratio_w = W / dst_w
    # ratio_h1 = H1 / dst_h
    # ratio_w1 = W1 / dst_w
    # for i in range(3):
    #     for h in range(dst_h):
    #         for w in range(dst_w):
    #             src_h = int(h * ratio_h1)
    #             src_w = int(w * ratio_w1)
    #             new_image1[h][w][i] = original_image1[src_h][src_w][i] / 255
    start = time.time()
    for i in range(3):
        for h in range(dst_h):
            for w in range(dst_w):
                src_h = int(h * ratio_h)
                src_w = int(w * ratio_w)
                new_image[h][w][i] = original_image[src_h][src_w][i]
    end = time.time()
    print(end-start)
    return new_image, original_image, new_image1


if __name__ == '__main__':
    image_path = './000028.jpg'
    new_image, original_image, new_image1 = nearest_interpolation(image_path=image_path, height=800, weight=600)
    # plt.imshow(new_image1)
    # plt.show()
    # original_image = cv2.imread(image_path)
    # cv2.imshow('', original_image)
    cv2.imshow('new image', new_image)
    cv2.waitKey(0)
    cv2.destroyWindow()
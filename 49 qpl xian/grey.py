import cv2
import numpy as np
import matplotlib.pyplot as plt


def norm_to_gray(image_path):
    '''
    input:original image
    method:mix pixel and mean pixel
    return:two gray image
    '''
    image = cv2.imread(image_path)
    # print(image.shape)
    # mixture_gray
    gray_image = np.array(image)
    gray_image[:, :, 0] = image[:, :, 0] * 0.11
    gray_image[:, :, 1] = image[:, :, 1] * 0.59
    gray_image[:, :, 2] = image[:, :, 2] * 0.3
    # np.sum(matrix, axis): sum mat based on axis
    gray_image = np.sum(gray_image, axis=2, dtype=np.uint8)
    # mean_gray
    gray_image2 = np.array(image) / 3
    gray_image2 = np.sum(gray_image2, axis=2, dtype=np.uint8)
    return gray_image, gray_image2


def norm_to_binary(gray_image):
    H, W = gray_image.shape
    binary_image = np.zeros((H, W))
    for h in range(H):
        for w in range(W):
            if gray_image[h][w]  < 127:
                binary_image[h][w] = 0
            else:
                binary_image[h][w] = 255
    return binary_image


if __name__ == '__main__':
    # use matplotlib to plot mutil_fig
        #step1: plt.figure()--to create fig
        #step2: plt.subplot()--ensure the loc of sub_fig and set:plt.title()
        #step3: plt.imshow()__to draw pic
        #step4: recur the step above
    image_path = './000028.jpg'
    image = plt.imread(image_path)
    gray_image, gray_image2 = norm_to_gray(image_path=image_path)
    binary_image = norm_to_binary(gray_image)
    # gray_image = gray_image.reshape(500, 375, 1)
    # print(gray_image.shape)
    plt.figure('gray image')
    plt.subplot(1, 3, 1)
    plt.title('original image')
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.title('mixture pixel')
    plt.imshow(gray_image, cmap='gray')# plt.imshow():take [h,w,3] as input, so we need param-(cmap='xxx')
    plt.subplot(1, 3, 3)
    plt.title('mean pixel')
    plt.imshow(gray_image2, cmap='gray')
    plt.show()
    plt.imshow(binary_image, cmap='gray')
    plt.show()
    # cv2.imshow('grey_image', grey_image) -----dtype=uint8
    # cv2.waitKey(0)
    # cv2.destroyWindow()


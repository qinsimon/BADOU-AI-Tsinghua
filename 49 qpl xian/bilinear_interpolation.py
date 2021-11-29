import numpy as np
import cv2
import time

'''
    running time: almost 10s!!! may i ask why???
'''


def bilinear_interpolation(image_path):
    original_image = cv2.imread(image_path)
    H, W, _ = original_image.shape
    new_image = np.zeros((800, 600, 3), dtype=np.uint8)
    ratio_h = H / 800
    ratio_w = W / 600
    start = time.time()
    for i in range(3):
        for h in range(800):
            for w in range(600):
                src_h = (h + 0.5) * ratio_h - 0.5
                src_w = (w + 0.5) * ratio_w - 0.5
                h0 = int(src_h)
                h1 = min(h0 + 1, H - 1)
                w0 = int(src_w)
                w1 = min(w0 + 1, W - 1)
                temp0 = (h1 - src_h) * original_image[h0, w0, i] + (src_h - h0) * original_image[h1, w0, i]
                temp1 = (h1 - src_h) * original_image[h0, w1, i] + (src_h - h0) * original_image[h1, w1, i]
                new_image[h, w, i] = int((w1 - src_w) * temp0 + (src_w - w0) * temp1)
    end = time.time()
    print(end-start)
    return new_image


if __name__ == '__main__':
    image_path = './000028.jpg'
    new_image = bilinear_interpolation(image_path)
    cv2.imshow('', new_image)
    cv2.waitKey(0)
    cv2.destroyWindow()
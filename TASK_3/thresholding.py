# MAXIMUM ENTROPY THRESHOLDING ALGORITHM FOR IMAGE SEGMENTATION

import cv2
import numpy as np
import matplotlib.pyplot as plt

def entp(x):
    temp = np.multiply(x, np.log(x))
    temp[np.isnan(temp)] = 0
    return temp

def max_entropy_threshold(image):

    # MAXIMUM ENTROPY FUNCTION
    H = cv2.calcHist([image], [0], None, [256], [0, 256])
    H = H / np.sum(H)

    a, b, c = np.zeros(256), np.zeros(256), np.zeros(256)

    for idx in range(1, 255):
        b[idx] = - np.sum(entp((H[:idx-1]) / np.sum(H[1:idx-1])))
        c[idx] = - np.sum(entp((H[idx:]) / np.sum(H[idx:])))
        a[idx] = b[idx] + c[idx]
    
    max = np.argmax(a)
    out = image > max
    out = out * 255
    
    return out


src_img = cv2.imread("Task_3/Image1.jpg")
gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

out = max_entropy_threshold(gray)

plt.plot(out)
plt.show()






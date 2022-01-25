import cv2
import numpy as np
from scipy import ndimage

def gauss_filter(img, size, sigma):
    """
    METHOD TO APPLY A GAUSSIAN FILTER OVER THE ORIGINAL IMAGE
    """
    return cv2.GaussianBlur(img, (size, size), sigma)

def sobel_operator(img):
    """
    METHOD TO CALCULATE SOBEL OPERATOR'S MAGNITUDE AND ORIENTATION 
    TO CALCULATE THE SOBEL EDGES ON THE ORIGINAL IMAGE
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Gx)
    Iy = ndimage.filters.convolve(img, Gy)

    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)

    return G, theta

def NMS(img, angle):
    N, M = img.shape

    empty = np.zeros((N, M), np.int32)
    theta = np.rad2deg(angle)

    for i in range(1, N-1):
        for j in range(1, M-1):
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                north = img[i, j+1]
                south = img[i, j-1]
            elif 22.5 <= theta[i, j] < 67.5:
                north = img[i+1, j-1]
                south = img[i-1, j+1]
            elif 67.5 <= theta[i,j] < 112.5:
                north = img[i+1, j]
                south = img[i-1, j]
            elif 112.5 <= theta[i,j] < 157.5:
                north = img[i-1, j-1]
                south = img[i+1, j+1]

            if img[i, j] >= north and img[i, j] >= south:
                empty[i, j] = img[i, j]
            else:
                empty[i, j] = 0
    
    return empty

def double_thresh(img, thresh):
    init_threshold = np.zeros(img.shape)
    low, high = thresh[0], thresh[1]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] < low:
                init_threshold[i, j] = 0
            elif img[i, j] >= low and img[i, j] < high:
                init_threshold[i, j] = 128
            else:
                init_threshold[i, j] = 255
    
    return init_threshold

def hysteresis(threshold_info):
    strong_pixels = np.zeros(threshold_info.shape)
    for i in range(0, threshold_info.shape[0]):
        for j in range(0, threshold_info.shape[1]):
            temp = threshold_info[i, j]
            if temp == 128:
                if threshold_info[i-1,j] == 255 or threshold_info[i+1,j] == 255 or threshold_info[i-1,j-1] == 255\
                or threshold_info[i+1,j-1] == 255 or threshold_info[i-1,j+1] == 255 or threshold_info[i+1,j+1] == 255\
                or threshold_info[i,j-1] == 255 or threshold_info[i,j+1] == 255:
                   strong_pixels[i, j] = 255
                elif temp == 255:
                    strong_pixels[i, j] = 255
                    
    return strong_pixels



raw_image = cv2.imread("image3.jpg", 0)
smooth_img = gauss_filter(raw_image, 5, 1)
G, theta = sobel_operator(smooth_img)
NMS_img = NMS(G, theta)
double_thresholding = double_thresh(NMS_img, [10, 20])
thresh_2_hyst = hysteresis(double_thresholding)


cv2.imshow("Original", raw_image)
cv2.imshow("FINAL IMAGE", thresh_2_hyst)
cv2.waitKey(0)
cv2.destroyAllWindows()
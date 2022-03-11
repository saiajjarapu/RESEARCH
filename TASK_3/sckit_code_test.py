import cv2
import numpy as np
from draw_contours import getContours
import matplotlib.pyplot as plt

def adap_thresh(input_img):
    h, w = input_img.shape

    S = w/8
    s2 = S/2
    T = 15.0

    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row, col] = int_img[0:row, 0:col].sum()

    out_img = np.zeros_like(input_img)

    for col in range(w):
        for row in range(h):
            y0 = max(row-s2, 0)
            y1 = min(row+s2, h-1)
            x0 = max(col-s2, 0)
            x1 = min(col+s2, w-1)

            count = (y1-y0)*(x1-x0)
            
            sum_ = int_img[y1, x1] - \
                   int_img[y0, x1] - \
                   int_img[y1, x0] + \
                   int_img[y0, x0]
            if input_img[row, col]*count < sum_*(100.-T)/100.:
                out_img[row, col] = 0
            else:
                out_img[row, col] = 255

    return out_img         

src_img = cv2.imread("TASK_2/Science_irhand.jpg")
hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

src_img = cv2.resize(src_img, (100,100))

print(f"Sized:: {src_img.shape}")

h, s, v = cv2.split(hsv)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
clahe_img = clahe.apply(v)

ret, glob_thresh = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(glob_thresh, (3, 3), 0)

adap_thresh = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY_INV, blockSize=1001, C=10)

adap = adap_thresh(clahe_img)

# getContours(adap_thresh, imgContour)

cv2.imshow("GRAY", v)
# plt.figure()
# plt.imshow(v)
# plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
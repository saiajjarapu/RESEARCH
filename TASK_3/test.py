import cv2
import numpy as np
from draw_contours import getContours

src_img = cv2.imread("TASK_3/Image1.jpg", 0)
resize_img = cv2.resize(src_img, (500,500))
resize_img = resize_img[100:400, 200:400]

imgContour = resize_img.copy()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
clahe_img = clahe.apply(resize_img)

ret, thresh1 = cv2.threshold(clahe_img, 100, 255, cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(thresh1, (3, 3), 0)

block_size = 401
constant = 135
thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, block_size, constant)

getContours(thresh2, imgContour)

cv2.imshow("OG", resize_img)
cv2.imshow("CLAHE", clahe_img)
# cv2.imshow("Thresh", thresh1)
# cv2.imshow("Adaptive", thresh2)
cv2.imshow("Results", imgContour)


cv2.waitKey()
cv2.destroyAllWindows()
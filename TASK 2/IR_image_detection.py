import cv2
from draw_contours import getContours
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread("TASK 2/Science_irhand.jpg")
imgContour = src_img.copy()
custom_kernel = np.array([[0, -1, 0], 
                          [-1, 5, -1],
                          [0, -1, 0]])
# imgblur = cv2.bilateralFilter(src_img, 9, 75, 75)
imgblur = cv2.filter2D(src_img, -1, custom_kernel)
imghsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
imgGray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

heat_min = np.array([5, 50, 50], np.uint8)
heat_max = np.array([30, 255, 255], np.uint8)
orange_filter = cv2.inRange(imghsv, heat_min, heat_max)

kernel = np.ones((5,5))
imgDilate = cv2.dilate(orange_filter, kernel=kernel)

imgCanny = cv2.Canny(imgblur, 100, 350, apertureSize=3)
edges = cv2.bitwise_and(imgDilate, imgCanny)


cv2.imshow("ORIGINAL", src_img)
cv2.imshow("BLURRED", imgblur)
cv2.imshow("GRAY SCALE", imgGray)
cv2.imshow("DILATED", imgDilate)
cv2.imshow("CANNY", imgCanny)
cv2.imshow("EDGES", edges)

cv2.waitKey()
cv2.destroyAllWindows()
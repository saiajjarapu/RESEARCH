import cv2
from draw_contours import getContours
import numpy as np

src_img = cv2.imread("TASK 2/Science_irhand.jpg")
imgContour = src_img.copy()
imghsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
imgGray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

heat_min = np.array([5, 50, 50], np.uint8)
heat_max = np.array([30, 255, 255], np.uint8)
orange_filter = cv2.inRange(imghsv, heat_min, heat_max)

kernel = np.ones((5,5))
imgDilate = cv2.dilate(orange_filter, kernel=kernel)

getContours(imgDilate, imgContour)
results = cv2.hconcat([src_img, imgContour])

cv2.imshow("FINAL", results)
cv2.waitKey()
cv2.destroyAllWindows()
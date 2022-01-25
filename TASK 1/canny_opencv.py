import cv2
import numpy as np
from draw_contours import getContours

# READ THE ORIGINAL IMAGE TO BE PROCESSED
img = cv2.imread("image2.jpg")
imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

imgCanny = cv2.Canny(imgGray, 100, 300)

kernel = np.ones((5,5))
imgDilate = cv2.dilate(imgCanny, kernel=kernel)
getContours(imgDilate, imgContour)

results = cv2.hconcat([img, imgContour])
cv2.imshow("RESULTS", results)
cv2.waitKey()

import cv2
from PIL import Image, ImageFilter
from draw_contours import getContours
import numpy as np

src_img = cv2.imread("TASK 2/Science_irhand.jpg")
imgContour = src_img.copy()

gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

imgGray = Image.fromarray(gray)
imghsv = Image.fromarray(hsv)

pixel_info = imghsv.load()

for i in range(imghsv.size[0]):
    for j in range(imghsv.size[1]):
        if (0, 35, 50) <= pixel_info[i, j] <= (30, 255, 255):
            pixel_info[i, j] = (255,255,255)
        else:
            pixel_info[i, j] = (0, 0, 0)

pil_image = np.array(imghsv)
cv_img = cv2.cvtColor(pil_image, cv2.COLOR_HSV2BGR)
img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

getContours(img_gray, imgContour)
results = cv2.hconcat([src_img, imgContour])
cv2.imshow("FINAL", results)
cv2.waitKey()
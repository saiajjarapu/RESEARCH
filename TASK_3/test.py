import cv2
import numpy as np
from draw_contours import getContours
import matplotlib.pyplot as plt

src_img = cv2.imread("TASK_3/Image1_mid.jpg", 0)
# resize_img = cv2.resize(src_img, (500,500))
# resize_img = resize_img[100:400, 200:400]

imgContour = src_img.copy()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
clahe_img = clahe.apply(src_img)

ret, glob_thresh = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(glob_thresh, (3, 3), 0)

block_size = 51
constant = 2
adap_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, block_size, constant)

getContours(adap_thresh, imgContour)

output = [src_img, glob_thresh, adap_thresh, imgContour]
titles = ['Original', ' Global Thresh', 'Adaptive Thresh', 'Result']

fig = plt.figure()
for i in range(len(output)):
    fig.add_subplot(1, len(output), i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])

fig.savefig('TASK_3/Results_Image2.png')
cv2.imwrite('TASK_3/Area_Img2.png', imgContour)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
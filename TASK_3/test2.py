import cv2
from cv2 import drawContours
import numpy as np
from draw_contours import getContours
import matplotlib.pyplot as plt
from PIL import Image

src_img = cv2.imread("TASK_2/Science_irhand.jpg")

correct_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2LUV)
imgContoor = correct_gray.copy()
array_img = np.asarray(correct_gray)
v = array_img[:,:,2]
# print(correct_gray)

# v[v<127] = 0
# v[v>127] = 255
# print(v)
# glob_thresh = cv2.threshold(v, 0.5, 1, cv2.THRESH_BINARY)

# print(f"Shape:: {v.shape}")

# getContours(v, imgContoor)

plt.figure()
plt.imshow(v)
plt.title("1")

# plt.figure()
# plt.imshow(array_img[:,:,0])
# plt.title("1")

# plt.figure()
# plt.imshow(array_img[:,:,1])
# plt.title("2")

# plt.figure()
# plt.imshow(array_img[:,:,2])
# plt.title("3")

# plt.figure()
# plt.imshow(gray)
# plt.title("4")

# plt.figure()
# plt.imshow(src_img[:,:,1], cmap='gray')
# plt.title("G channel")
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
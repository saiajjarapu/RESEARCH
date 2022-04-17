import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from draw_contours import getContours
import matplotlib.pyplot as plt
import os

# EXTRACT CURRENT WORKING DIRECTORY
cwd = os.getcwd()
WORK_DIR = os.path.join(cwd, "TASK_3")
img_path = os.path.join(WORK_DIR, "Image2.jpg") # CHANGE THE SECOND PARAMETER TO SELECT THE IMAGE

# READ THE ORIGINAL IMAGE TO BE PROCESSED
src_img = cv2.imread(img_path, 0)
resize_img = cv2.resize(src_img, (500,500)) # RESIZE THE ORIGINAL IMAGE
resize_img = resize_img[100:400, 200:400]
imgContour = resize_img.copy()

# PASS THE ORIGINAL IMAGE TO CLAHE TO EQUALIZE THE IMAGE PIXEL VALUES
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
clahe_img = clahe.apply(resize_img)

# PASS THE CLAHE IMAGE THRU FIRST PASS GLOABL THRESHOLDING AND BLUR 
ret, glob_thresh = cv2.threshold(clahe_img, 115, 255, cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(glob_thresh, (5, 5), 0)

# PASS THE GLOBALLY THRESHOLDED IMAGE THRU ADAPTIVE THRESHOLD TO FURTHER ENHANCE THE AREAS OF INTEREST
block_size = 51
constant = 2
adap_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, block_size, constant)

# FUNCTION TO CALCULATE THE AREA OF THE FOUND OBJECTS
getContours(adap_thresh, imgContour)

# PLOTTING THE RESULTS
output = [src_img, glob_thresh, adap_thresh, imgContour]
titles = ['Original', ' Global Thresh', 'Adaptive Thresh', 'Result']
fig = plt.figure()
for i in range(len(output)):
    fig.add_subplot(1, len(output), i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])

# SAVING THE OUTPUT IMAGES
filename = "Final_Image_Image2.jpg" # CHANGE TO DESIRED OUTPUT IMAGE FILENAME
save_dir = "TASK_3/" + filename
fig.savefig(save_dir)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from draw_contours import getContours
import matplotlib.pyplot as plt
import os

# EXTRACT CURRENT WORKING DIRECTORY
cwd = os.getcwd()
WORK_DIR = os.path.join(cwd, "TASK_1")
img_path = os.path.join(WORK_DIR, "image4.jpg") # CHANGE THE SECOND PARAMETER TO SELECT THE IMAGE

# READ THE ORIGINAL IMAGE TO BE PROCESSED
img = cv2.imread(img_path)
imgContour = img.copy()

# IMAGE BLURRING, COLOR CONVERSION, CANNY EDGE DETECTION
imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray, 100, 300)

# DILATE THE CANNY EDGES TO CREATE A MASK
kernel = np.ones((5,5))
imgDilate = cv2.dilate(imgCanny, kernel=kernel)
getContours(imgDilate, imgContour) # FUNCTION TO CALCULATE THE AREA OF THE FOUND OBJECTS

# PLOTTING THE RESULTS
output = [img, imgDilate, imgContour]
titles = ["Original", "Mask", "Final Objects"]
fig = plt.figure()
for i in range(len(output)):
    fig.add_subplot(1, len(output), i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])

# SAVING THE OUTPUT IMAGES
filename = "Final_Image2.jpg" # CHANGE TO DESIRED OUTPUT IMAGE FILENAME
save_dir = "TASK_1/" + filename
fig.savefig(save_dir)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()

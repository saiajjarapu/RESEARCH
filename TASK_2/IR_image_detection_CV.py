import warnings
warnings.filterwarnings("ignore")
import cv2
from draw_contours import getContours
import numpy as np
import matplotlib.pyplot as plt
import os

# EXTRACT CURRENT WORKING DIRECTORY
cwd = os.getcwd()
WORK_DIR = os.path.join(cwd, "TASK_2")
img_path = os.path.join(WORK_DIR, "Science_irhand.jpg") # CHANGE THE SECOND PARAMETER TO SELECT THE IMAGE

# READ THE ORIGINAL IMAGE TO BE PROCESSED
src_img = cv2.imread(img_path)
imgContour = src_img.copy()

# CONVERTING SOURCE IMAGE TO HSV AND GRAY COLOR SPACES
imghsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
imgGray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# EXTRACTING THE LOWER AND UPPER THRESHOLD VALUES FOR THE AREAS OF INTEREST
heat_min = np.array([5, 50, 50], np.uint8)
heat_max = np.array([30, 255, 255], np.uint8)
# CREATING COLOR MASK, IN THIS CASE THE COLOR IS ORANGE
orange_filter = cv2.inRange(imghsv, heat_min, heat_max)

# DILATING THE CREATED COLOR MASK
kernel = np.ones((5,5))
imgDilate = cv2.dilate(orange_filter, kernel=kernel)

# FUNCTION TO CALCULATE THE AREA OF THE FOUND OBJECTS
getContours(imgDilate, imgContour)

# PLOTTING THE RESULTS
output = [src_img, imgContour]
titles = ["Original", "Final Objects"]
fig = plt.figure()
for i in range(len(output)):
    fig.add_subplot(1, len(output), i+1)
    plt.imshow(output[i], cmap="gray")
    plt.title(titles[i])

# SAVING THE OUTPUT IMAGES
filename = "Final_Image.jpg" # CHANGE TO DESIRED OUTPUT IMAGE FILENAME
save_dir = "TASK_2/" + filename
fig.savefig(save_dir)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()

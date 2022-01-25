import cv2
import numpy as np
from scipy import ndimage

class Canny_edges():
    def __init__(self, img, size, sigma, thresh):
        self.img = cv2.imread(img, 0)
        self.size = size
        self.sigma = sigma
        self.low_thresh = thresh[0]
        self.high_thresh = thresh[1]
        self.final_image = None

    def gauss_filter(self):
        """
        METHOD TO APPLY A GAUSSIAN FILTER OVER THE ORIGINAL IMAGE
        """
        # TODO: TROUBLESHOOT ERROR "File "c:\Users\sai\Documents\Masters\Spring 2022\Research\example.py", line 107, in <module>   
            #     test.final()
            #   File "c:\Users\sai\Documents\Masters\Spring 2022\Research\example.py", line 89, in final       
            #     self.gauss_filter()
            #   File "c:\Users\sai\Documents\Masters\Spring 2022\Research\example.py", line 18, in gauss_filter
            #     self.filtered_image = cv2.GaussianBlur(self.img, (self.size, self.size), self.sigma)
            # cv2.error: OpenCV(4.5.5) :-1: error: (-5:Bad argument) in function 'GaussianBlur'"
        self.filtered_image = cv2.GaussianBlur(self.img, (self.size, self.size), self.sigma)
        
        return self.filtered_image

    def sobel_operator(self):
        """
        METHOD TO CALCULATE SOBEL OPERATOR'S MAGNITUDE AND ORIENTATION 
        TO CALCULATE THE SOBEL EDGES ON THE ORIGINAL IMAGE
        """
        self.Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        self.Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        self.Ix = ndimage.filters.convolve(self.img, self.Gx)
        self.Iy = ndimage.filters.convolve(self.img, self.Gy)

        self.G = np.hypot(self.Ix, self.Iy)
        self.G = self.G / self.G.max() * 255
        self.theta = np.arctan2(self.Iy, self.Ix)

        return self.G, self.theta

    def NMS(self):
        self.n, self.m = self.img.shape
        self.empty = np.zeros((self.n, self.m), np.int32)

        self.angle = np.rad2deg(self.theta)
        self.angle[self.angle < 0] += 180

        for i in range(1, self.n-1):
            for j in range(1, self.m-1):
                if 0 <= self.angle[i, j] < 22.5 or 157.5 <= self.angle[i, j] <= 180:
                    north = self.img[i, j+1]
                    south = self.img[i, j-1]
                elif 22.5 <= self.angle[i, j] < 67.5:
                    north = self.img[i+1, j-1]
                    south = self.img[i-1, j+1]
                elif 67.5 <= self.angle[i,j] < 112.5:
                    north = self.img[i+1, j]
                    south = self.img[i-1, j]
                elif 112.5 <= self.angle[i,j] < 157.5:
                    north = self.img[i-1, j-1]
                    south = self.img[i+1, j+1]

                if self.img[i, j] >= north and self.img[i, j] >= south:
                    self.empty[i, j] = self.img[i, j]
                else:
                    self.empty[i, j] = 0

        return self.empty
    
    def thresh(self):
        self.resolution = np.zeros((self.n, self.m), np.int32)

        self.weak_pixels = np.int32(self.low_thresh)
        self.strong_pixels = np.int32(self.high_thresh)

        self.high_i, self.high_j = np.where(self.img >= self.high_thresh)
        self.zero_i, self.zero_j = np.where(self.img < self.low_thresh)
        self.low_i, self.low_j = np.where(self.img <= self.high_thresh and self.img >= self.low_thresh)

        self.resolution[self.high_i, self.high_j] = self.strong_pixels
        self.resolution[self.low_i, self.low_j] = self.weak_pixels

        return self.resolution

    def hysteresis(self):
        """
        return: variable with final image
        """
        #TODO: FINISH THIS FUNCTION
        pass
    
    def final(self):
        self.gauss_filter()
        self.sobel_operator()
        self.NMS()
        self.thresh()
        self.hysteresis()
        cv2.imshow("ORIGINAL", self.img)
        cv2.imshow("FINAL EDGES", self.final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_area(self):
        """
        return: print statements with area of the contour &
                bounding boxes around the contour
        """
        #TODO: FINISH THIS FUNCTION
        pass

filepath = "image1.jpg"
test = Canny_edges(filepath, 3, 1, [75, 255])
test.final()



import cv2
import numpy as np
import time

img = cv2.imread('./lena.jpg', 1)

cv2.imshow("Original Lena", img)
for i in range(1,25,2):
    blur_kernel = np.ones((i, i), np.float32)/(i**2)
    #This function convolves an image with kernel
    #blurred_img = cv2.filter2D(img, -1, blur_kernel)
    blurred_img = cv2.medianBlur(img, (i))
    cv2.imshow("Blurred Lena", blurred_img)
    cv2.waitKey(0)
avg_blur = cv2.blur(img, (5, 5))
# it takes gaussian kernel and computes convolution
# the sigmaX and sigmaY are also provided which
# are values of standard deviation in X and Y direction
gaus_blur = cv2.GaussianBlur(img, (5, 5), 0, 0)
# It takes median, so calculated value is the one from current image only
# it is highly effective for salt-and-pepper noise
median_blur = cv2.medianBlur(img, (5))
# This filter takes one more gaussian filter of intensity difference so that
# only those pixels with similar intensity to central pixel are considered
# for blurring and thereby preserves edges
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

cv2.imshow("Average Kernel", avg_blur)
cv2.imshow("Gaussian Kernel", gaus_blur)
cv2.imshow("Median Kernel", median_blur)
cv2.imshow("Bilateral Kernel", bilateral_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

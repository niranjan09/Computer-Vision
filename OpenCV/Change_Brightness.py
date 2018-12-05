import cv2
import numpy as np
import time

img = cv2.imread('./lena.jpg', 1)
"""
we are increasing the brightness by adding into each rgb value of pixels
value cannot go aboe 255 and below 0 so either use cv functions or else
take care of this if doing manually using np
"""
cv2.namedWindow("Normal Image", cv2.WINDOW_NORMAL)
cv2.resize("Normal Image", 600, 600)
print img[1,1]
# brighter_img = img + 50 : this will not work as care of limits 0 and 255
# will not be taken so use brighter_img = np.minimum(img+50, 255)
# that too if img is more than 8 bit
M = np.ones(img.shape, dtype = "uint8")
brighter_img = cv2.add(img, M*50)
#brighter_img = img +50
#brighter_img = np.maximum(255*(brighter_img > img), brighter_img)
cv2.namedWindow("Brighter Image", cv2.WINDOW_NORMAL)

# darker_img = img  - 50 will not work for the same reason mentioned above
darker_img = cv2.subtract(img, M)
cv2.namedWindow("Darker Image", cv2.WINDOW_NORMAL)

cv2.imshow("Normal Image", img)
cv2.imshow("Brighter Image", brighter_img)
cv2.imshow("Darker Image", darker_img)
# To increase the brightness of image after every 100 ms
#for i in range(200):
#    cv2.imshow("Brighter Image", cv2.add(img, M*i))
#    cv2.waitKey(100)

while True:
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break

cv2.imwrite("brighter_lena.jpg", brighter_img)
cv2.imwrite("darker_lena.jpg", darker_img)

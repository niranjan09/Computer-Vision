import cv2
import numpy as np
import time

font = cv2.FONT_HERSHEY_SIMPLEX

def showRGB(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        rgb_string = str(img[y, x][::-1])
        print rgb_string, x, y
        cv2.putText(img, rgb_string, (200, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

img = cv2.imread('./DinoGame67percentZoom.png', 1)
cv2.namedWindow("Start Position")
cv2.setMouseCallback("Start Position", showRGB)
while True:
    cv2.imshow("Start Position", img)
    key_pressed = cv2.waitKey(1)
    if(key_pressed == ord('q')):
        break

cv2.destroyAllWindows()

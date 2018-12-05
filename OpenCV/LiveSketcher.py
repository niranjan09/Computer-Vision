import cv2
import numpy as np

vc = cv2.VideoCapture(0)

while True:
	ret, frame = vc.read()
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cleaned_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
	canny_edged_image = cv2.Canny(cleaned_image, 10, 25)
	#invert binarized image
	ret, inv_frame = cv2.threshold(canny_edged_image, 10, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow("Live Sketcher", inv_frame)
	key_pressed = cv2.waitKey(1)
	if(key_pressed == ord('q')):
		break

vc.release()
cv2.destroyAllWindows()

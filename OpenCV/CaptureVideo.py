import cv2

vcap = cv2.VideoCapture('http://192.168.1.102:8080/')

while(vcap.isOpened()):
	ret, frame = vcap.read()
	cv2.imshow('VIDEO', frame)
	cv2.waitKey(1)
	key_pressed = cv2.waitKey(1)
	if(key_pressed == ord('q')):
		break


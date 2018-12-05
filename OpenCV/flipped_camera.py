import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 300)
cap.set(4, 300)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('flipped', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 300, 300)
cv2.resizeWindow('flipped', 300, 300)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow("flipped", np.fliplr(gray)) #flipud, rot90
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

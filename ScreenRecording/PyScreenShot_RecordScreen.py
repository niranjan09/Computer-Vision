import pyscreenshot as ImageGrab
import cv2
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (700, 200))

# grab fullscreen
a=0
temp = []
time.sleep(3)
start_time = time.time()
print("Started Recording", start_time)
while True:
    im = ImageGrab.grab(bbox = (300, 300, 1000, 500))
    im = np.array(im.convert('RGB'))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    print(time.time() - start_time)
    #cv2.imshow("ScreenCast", im)
    temp.append(im)
    a+=1
    if a>100:
        break
    # show image in a window
end_time = time.time()
print("Done Recording", end_time, "Total time:", end_time - start_time)
for i in temp:
    cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    out.write(i)
out.release()

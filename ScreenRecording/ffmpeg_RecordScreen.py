# Python script to read video frames and timestamps using ffmpeg
import subprocess as sp
import threading

import matplotlib.pyplot as plt
import numpy
import cv2

ffmpeg_command = [ 'ffmpeg',
                   '-nostats', # do not print extra statistics
                    #'-debug_ts', # -debug_ts could provide timestamps avoiding showinfo filter (-vcodec copy). Need to check by providing expected fps TODO
                    '-r', '25', # output 30 frames per second
                    '-f', 'x11grab',
                    '-video_size', '1366x768',
                    '-i', ':0.0+0,0',
                    '-an','-sn', #-an, -sn disables audio and sub-title processing respectively
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    #'-vcodec', 'copy', # very fast!, direct copy - Note: No Filters, No Decode/Encode, no quality loss
                    #'-vframes', '20', # process n video frames only. For Debugging
                    '-vf', 'showinfo', # showinfo videofilter provides frame timestamps as pts_time
                    '-f', 'image2pipe', 'pipe:1' ] # outputs to stdout pipe. can also use '-' which is redirected to pipe


# seperate method to read images on stdout asynchronously
def AppendProcStdout(proc, nbytes, AppendList):
    while proc.poll() is None: # continue while the process is alive
        AppendList.append(proc.stdout.read(nbytes)) # read image bytes at a time

# seperate method to read image info. on stderr asynchronously
def AppendProcStderr(proc, AppendList):
    while proc.poll() is None: # continue while the process is alive
        try: AppendList.append(proc.stderr.next()) # read stderr until empty
        except StopIteration: continue # ignore stderr empty exception and continue


if __name__ == '__main__':
    # run ffmpeg command
    pipe = sp.Popen(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE)

    # 2 threads to talk with ffmpeg stdout and stderr pipes
    framesList = [];
    frameDetailsList = []
    # assuming rgb video frame with size 1366x768
    appendFramesThread = threading.Thread(group=None, target=AppendProcStdout, name='FramesThread', args=(pipe, 1366*768*3, framesList), kwargs=None, verbose=None)
    appendInfoThread = threading.Thread(group=None, target=AppendProcStderr, name='InfoThread', args=(pipe, frameDetailsList), kwargs=None, verbose=None)

    # start threads to capture ffmpeg frames and info.
    appendFramesThread.start()
    appendInfoThread.start()

    # wait for few seconds and close - simulating cancel
    import time; time.sleep(10)
    pipe.terminate()

    # check if threads finished and close
    appendFramesThread.join()
    appendInfoThread.join()

    # save an image per 30 frames to disk
    savedList = []
    for cnt,raw_image in enumerate(framesList):
        if (cnt%30 != 0): continue
        image1 =  numpy.fromstring(raw_image, dtype='uint8')
        image2 = image1.reshape((768,1366,3))  # assuming rgb image with size 1280 X 720
        # write video frame to file just to verify
        videoFrameName = 'video_frame{0}.png'.format(cnt)
        cv2.imwrite(videoFrameName,cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY))
        savedList.append('{} {}'.format(videoFrameName, image2.shape))
        #cv2.imshow("Frame",image2)
        #cv2.waitKey(1000)
"""
    print '### Results ###'
    print 'Images captured: ({}) \nImages saved to disk:{}\n'.format(len(framesList), savedList) # framesList contains all the video frames got from the ffmpeg
    print 'Images info captured: \n', ''.join(frameDetailsList) # this contains all the timestamp details got from the ffmpeg showinfo videofilter and some initial noise text which can be easily removed while parsing
"""

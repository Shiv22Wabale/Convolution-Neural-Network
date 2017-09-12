import cv2
import sys
import os
from random import randint


os.system("rm capture/*")
vidcap = cv2.VideoCapture(sys.argv[1])
vidcap.set(0, randint(10, 200) )      # just cue to 200 msec. position
success,image = vidcap.read()
count = 0
success = True
while success:
	success,image = vidcap.read()
	print 'Read a new frame: ', success
	if success :
		cv2.imwrite("capture/frame%d.jpg" % count, image)     # save frame as JPEG file
	count += 1
if count > 0 :
	os.system("python detect.py haarcascade_frontalface_default.xml capture/")

# import the necessary packages
import random
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from PIL import Image
# construct the argument parser and parse the arguments

COLORS = [(139, 0, 0), 
          (0, 100, 0),
          (0, 0, 139)]

def random_color():
    return random.choice(COLORS)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream

_,f = camera.read()
f = imutils.resize(f, width=500)
gray1 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
avg = np.float32(gray1)

currFrameIndex = 0;
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break	

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	cv2.accumulateWeighted(gray,avg,0.01)
	
	resСonvertScaleAbs = cv2.convertScaleAbs(avg)

	# compute the absolute difference between the current frame and
	# average frame
	frameDelta = cv2.absdiff(resСonvertScaleAbs, gray)
	thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
	
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
	
	blank_image = np.zeros((500,500,3), np.uint8)
	blank_image[:] = (255, 255, 255)
	# loop over the contours
	
	if currFrameIndex > 150:

		i = 0
		for c in cnts:
			# if the contour is too small, ignore it
			
			cv2.drawContours(blank_image, [c], 0, random_color(), 3)
		
			if cv2.contourArea(c) < args["min_area"]:
				continue

			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)		
			i = i + 1

	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	cv2.imshow("Сountors", blank_image)
	
	imWithRect = Image.fromarray(frame)
	countorsIm = Image.fromarray(blank_image)
	
	imWithRect.save("result_img/frame" + str(currFrameIndex) +".jpeg")	
	countorsIm.save("result_img/thresh" + str(currFrameIndex) +".jpeg")
	
	currFrameIndex = currFrameIndex + 1
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
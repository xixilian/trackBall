import numpy as np
import cv2
import sys
import imutils

#cap = cv2.VideoCapture('./output.avi')
cap = cv2.VideoCapture('./res.avi')
# Exit if video not opened.
if not cap.isOpened():
	print "Could not open video"
        sys.exit()


#feature_params = dict(maxCorners=100, qualityLevel=0.6, minDistance=25, blockSize=9)
#kernel = np.ones((5,5), np.uint8)

# background subtractor
#fgbg = cv2.createBackgroundSubtractorMOG2()


#balls = []
counter = 0

#write video
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

while True  :

	ret, frame = cap.read()

	#print(frame.shape)
	if not ret:
        	print 'Cannot read video file'
        	sys.exit()
	#resized = imutils.resize(frame, width=128)
    	#frame = resized	


	counter = counter +1	
	cv2.putText(frame, "{} th frame".format(counter),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)	
 
	cv2.imshow('img', frame)
	#mask = fgmask
	#resized = cv2.resize(mask, (640,480))
	#mask = fgmask
	#print (resized.shape)	
	
	cv2.imwrite("./frames/%03d.jpg" % counter,frame)  
	     
	k=cv2.waitKey(1) & 0xff
	if k == 27 : break

cap.release()

print(counter)

cv2.destroyAllWindows()


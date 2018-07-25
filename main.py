import numpy as np
import cv2
import sys
import imutils
import os
import glob
from collections import deque

# import the necessary packages

from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

# import moduls
import color_det as det
import run

corners = None

obstacles = []
stuffs = deque(maxlen = 64)

surfaces = []
normals = []

goal = None

sort = False

# files for debugging
f = open('./ptscoordinate.txt','w')
f1 = open('./normalscoordinate.txt','w')
f2 = open('./obstaclecoordinate.txt','w')

def calculate_normal(pts):

	p1 = pts[0]
	p2 = pts[1]
	p3 = pts[2]

	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]

		
	angle = np.arctan2(dx,dy)
	len = 80
    
	x1 = (p1[0] + p2[0])/2.0
	y1 = (p1[1] + p2[1])/2.0
	x2 = x1 + len * np.cos(angle)
	y2 = y1 - len * np.sin(angle)
 
	return ([(x1,y1),(x2,y2)])

counter = 0

# put the image files in the folder frames, numbered them
for img in sorted(glob.glob("./frames/*.jpg"))  :
	
	frame = cv2.imread(img)
        
	if corners is None:
        	corners = run.find_corners(frame)

	if (len(stuffs) < 1):
	        stuffs = det.detect_env(frame,corners,f2)

	if (len(normals) < 1) :
		for (i, rect) in enumerate(stuffs):
			if (i > 0):
		    		pts = calculate_normal(rect[:3])
				normals.append(pts)
				obstacles.append(rect)
				f1.write(','.join( str(p) for p in pts))
				f1.write('\n')

	if ((len(obstacles) == 3) and not sort) :
		# sort with the y coordinate of the top left element in rectangle
		obstacles.sort(key=lambda x: x[0][1])
		sort = True

	if ( len(surfaces) < 1  and len(obstacles) == 3):
		for o in obstacles:
			f2.write('obstacle' + '\n')
			f2.write(','.join(str(v) for v in o))
			f2.write('\n')
			surfaces.append(o[:2])

	# draw goal and obstacles :
	frame = run.track(frame, corners, counter, surfaces,f)
	for (i, rect) in enumerate(stuffs):
   
	        color = 'blue'
	        text = 'obstacle'
	        if (i == 0):
		    color = 'violet'
		    text = 'goal'

		box = cv2.minAreaRect(rect)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		cv2.drawContours(frame, [box], -1, det.colors[color], 2)

		for (x, y) in rect:
		    cv2.circle(frame, (int(x), int(y)), 5, det.colors[color], -1)

		if (i > 0):
		    pts = normals[i-1]
		    #print (pts)
		    cv2.arrowedLine(frame, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), det.colors['green'], 2)

		cv2.putText(frame, text, (int(rect[0][0] + 15), int(rect[0][1] + 15)),
				    					cv2.FONT_HERSHEY_SIMPLEX, 0.55, det.colors[color], 2)
	
			

	
	counter +=1
	cv2.imshow('Frames', frame)
	cv2.imwrite("./results/%03d.jpg" % counter,frame)  
	key = cv2.waitKey(2) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
        	break

cv2.destroyAllWindows()
f.close()
f1.close()
f2.close()








# import the necessary packages
from __future__ import print_function
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
#import urllib #for reading image from URL
import glob

# import the necessary packages

from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

import run

def order_points(pts, corners):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    ySorted = pts[np.argsort(pts[:, 1]), :]

    upper = ySorted[-1]
    
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    left = leftMost[0][0] 
    right = rightMost[1][0]
    
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
 

# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(100, 40, 80), 'yellow':(23, 59, 119),  'violet':(130,25,30), 'brown' : (10,60,10)} #assign new item lower['blue'] = (93, 10, 0) 'orange':(0, 50, 80),
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(120,255,255), 'yellow':(54,255,255),  'violet':(160,200,90), 'brown' : (20,75,30)} #'orange':(20,255,255)
 
# define standard colors for circle around the object
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217),  'violet':(124,93,153), 'brown' : (15,70,20)} # 'orange':(0,140,255),


def detect_shape(c):
	# initialize the shape name and approximate the contour
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	print (len(approx))
 
	# if the shape has 4 vertices, it is either a square or
	# a rectangle
	# or 2 , is the violet with blue dots on it
	if len(approx) == 4:
		shape = "rectangle"
 
	# if the shape is a pentagon, it will have 5 vertices
	elif len(approx) >= 6:
		shape = "concave"

	return shape

corners =None

counter = 0



def detect_env(frame, corners,f):

	stuffs = deque(maxlen = 64)

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


        kernel = np.ones((9,9),np.uint8)

	mask = cv2.inRange(hsv, lower['blue'], upper['blue'])
    	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        '''   
	maskv = cv2.inRange(hsv, lower['violet'], upper['violet'])
	maskv = cv2.morphologyEx(maskv, cv2.MORPH_OPEN, kernelv)
	maskv = cv2.morphologyEx(maskv, cv2.MORPH_CLOSE, kernelv)
	   '''
        
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
	#mask= cv2.bitwise_or(maskb,maskv)
	#cv2.imshow("mask blue", maskb)
	cv2.imshow("mask", mask)

        '''
        mask =  cv2.inRange(hsv, lower['brown'], upper['brown'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelv)
        '''

        cv2.imshow("mask", mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        	    cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

       
        	# only proceed if at least one contour was found
        if len(cnts) > 0:
        	    
            		# loop over the contours individually
        	for (i, c) in enumerate(cnts):
                	# if the contour is not sufficiently large, ignore it
                	if cv2.contourArea(c) < 500:                    		
				        continue

				
				
                	# compute the rotated bounding box of the contour, then
                	# draw the contours
                	box = cv2.minAreaRect(c)
                	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else 								cv2.boxPoints(box)
                	box = np.array(box, dtype="int")
			
			# order the points in the contour such that they appear
                	# in top-left, top-right, bottom-right, and bottom-left
                	# order, then draw the outline of the rotated bounding
                	# box
                	rect = order_points(box, corners)

                        if rect is None:
				            continue

                	cv2.drawContours(frame, [box], -1, colors['blue'], 2)
                	# show the original coordinates
                	#print("Object #{}:".format(i + 1))
                	#print(box)

                	#get shape, rectangles are obstacles, the concave one is the goal
			shape = detect_shape(c)

			# luckily the goal has a specific shape
			if (shape == 'concave'):
				stuffs.appendleft(rect)
                        	f.write('goal' + '\n')
                        	f.write( ','.join( str(v) for v in rect))
                        	f.write('\n') 
			        # the rest are ostacles
                    	else :
				stuffs.append(rect)

                	# show the re-ordered coordinates
                	#print(rect.astype("int"))
                	#print("")
                	# loop over the original points and draw them

			'''
                	for ((x, y), color) in zip(rect, colors):
                    	cv2.circle(frame, (int(x), int(y)), 5, colors[key], -1)

                	# draw the object num at the top-left corner
                	cv2.putText(frame, key +" "+ shape,
                                (int(rect[0][0] - 15), int(rect[0][1] - 15)),	
                   			cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[key], 2)	
			'''
	return stuffs	



import numpy as np
import cv2
import sys
import imutils
import os
import glob

kernel = np.ones((5,5), np.uint8)

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()


cnt_min_vertices = 0
cnt_max_vertices = 1000000
cnt_solidity = [0.0, 100.0]
cnt_min_area = 200.0
cnt_max_area = 5000.0


points = []
radiuses = []


def find_corners(img):

	gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
	
	ret,gray = cv2.threshold(gray, 85,255,1)
	
	cv2.imshow('filter', gray)
	contours = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]

	tmp_x = []
	tmp_y = []
	results = []
	
	if len(contours) > 0 : 

		c = max(contours, key=cv2.contourArea)
		x,y,w,h = cv2.boundingRect(c)
		
        	results.append(x)
		results.append(y)
		results.append(w)
		results.append(h)

	print results        	
    
	return results


def track(frame, corners, counter, surfaces,f):		

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	
    cv2.imshow('mask',fgmask)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel2 = np.ones((1,1),np.uint8)
    mask2 = cv2.inRange(hsv, (0, 50, 50), (10,255,255))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel2)
    
    mask3 = cv2.inRange(hsv, (166, 50, 70), (180,255,255))
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel2)
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel2)

    mask_r= cv2.bitwise_or(mask3,mask2)
    cv2.imshow('red mask', mask_r)
	
    res = cv2.bitwise_and(fgmask,mask_r)

    cnts = cv2.findContours(mask_r.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
		
    detected_balls = cv2.goodFeaturesToTrack(res.copy(), maxCorners=100, qualityLevel=0.5,minDistance=20, blockSize=9)


	# do it once
    if corners is None:
        corners = find_corners(frame)

    if len(cnts) > 0:
	
        for cnt in cnts :
	
            ((u, v), radius) = cv2.minEnclosingCircle(cnt)
			
            area = cv2.contourArea(cnt)
            if (area > cnt_max_area or area < cnt_min_area):
                continue

            center = (int(u),int(v))
            
            cv2.circle(frame,(int(u),int(v)), int(radius), (0,0,255),2)
            points.append((int(u),int(v)))
            f.write(str(int(u)) + ' , ' + str(int(v)) + '\n')
            radiuses.append(radius)
	

    s = 'undefined'

	# decide status, sliding, bouncing, falling
    if ((len(points) > 5) and (len(surfaces) > 0) and (len(radiuses) > 0) ) :
		
        frame = get_status(points, frame, surfaces, np.mean(radiuses),f )

    for i in np.arange(1, len(points)):
		# if either of the tracked points are None, ignore them
        if points[i - 1] is None or points[i] is None:
            continue
 		
		# check to see if enough points have been accumulated in the buffer
		#if i == 1 and points[-10] is not None:
        cv2.circle(frame, points[i], 2, (20, 255, 255), 1)
        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 1)

    counter = counter +1
    #cv2.putText(frame, "{} th frame".format(counter),
    #	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow('img', frame)
    #mask = fgmask
    #resized = cv2.resize(mask, (640,480))
    #mask = fgmask
    #print (resized.shape)
    #out1.write(mask)
    cv2.imwrite("./color/%03d.jpg" % counter,frame)
    cv2.imwrite("./res/%03d.jpg" % counter,mask)

    return frame



def get_status(pts, frame, surfaces, r,f):
	
	# it is asserted that pts has at least length 5
	# get last 3 points
	p1 = pts[-1]
	p2 = pts[-2]
	p3 = pts[-3]

	surface = find_surface(p1, surfaces)

	#print(p1)
	#print(surface)
	
	dx1 = p1[0] - p2[0]
	dy1 = p1[1] - p2[1]

	dx2 = p2[0] - p3[0]
	dy2 = p2[1] - p3[1]
	
	angle1 = np.arctan2(dy1,dx1)* 180/np.pi
	angle2 = np.arctan2(dy2,dx2)* 180/np.pi

	s = 'slow'
	diff = abs(angle1 - angle2)

	y =p1[1] + r

	distance = np.cross(surface[1] - surface[0], surface[0] - p1)/np.linalg.norm(surface[1] - surface[0])

	if not (dy1 == 0 and dx1 == 0):
		if (diff < 15):
			if ((dy1 > 10 or dy1 < 0) and (r <= distance)):
				s = 'air'
			else :
				s = 'contact'
		elif(dy1 > 1 ):

			if (r < distance):
				s = 'air'
			else :
				s = 'contact'
		elif(diff >= 90 and dy1 < -2) :
			s = 'bouncing'

		if ((distance < 0) and dy1 > 0):
			if (s == 'contact'):
				s = 'air'

	f.write(s)
	f.write('\n')

	cv2.putText(frame, "{} degree".format(diff),
			(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 1)

	cv2.putText(frame, "{} y diff".format(dy1),
			(10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 1)
	cv2.putText(frame, s,
			(10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 200), 2)	

	cv2.putText(frame, "{} distance to surface".format(distance), 
			(10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 1)
	
	cv2.putText(frame, "{} y + r".format(y), 
			(10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 1)

	cv2.putText(frame, "{} surface".format(surface), 
			(10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 1)		
	return frame
	

def find_surface(p, surfaces):
	
	for s in surfaces :
		y_sorted = s[np.argsort(s[:, 1]), :]
		y_max = y_sorted[-1][1]
	
		if p[1] < y_max:
			return s


	return surfaces[-1]



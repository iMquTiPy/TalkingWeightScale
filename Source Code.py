from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import time
import numpy as np

def bird_view(image):
	image = imutils.resize(image, height=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5,5), 0)
	edged = cv2.Canny(blurred, 200, 92, 255)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	displayCnt = None

	# loop over the contours
	count=0
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if the contour has four vertices, then we have found
		# the display
		if len(approx) == 4:
			displayCnt = approx
			count=1
			break
	if count == 1:
		warped = four_point_transform(gray, displayCnt.reshape(4, 2))
		output = four_point_transform(image, displayCnt.reshape(4, 2))
		return count, warped, output
	return 0, None, None

#dictionary defineD according to : 
# 	  _0_
#   1|	   |2
#    |_3_|	
#   4|	   |5
#    |_6_|
		  	
DIGITS_LOOKUP = {					#dictionary 
	(1, 1, 1, 0, 1, 1, 1): 0,		# tuple = key, ans = value
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9,
    (0, 0, 1, 0, 0, 1, 1): 10,
    (0, 0, 0, 0, 0, 0, 0): 100,
    (1, 0, 1, 1, 0, 0, 1): 30,
    (0, 1, 1, 1, 0, 1, 1): 90,
    (1, 1, 0, 0, 1, 1, 1): 100,
    (1, 0, 0, 1, 1, 1, 1): 60
}

# load the example image


#image = cv2.imread('..\\Images_to_test\\9.jpeg')
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
cap = cv2.VideoCapture(0)

while True:

	image = cap.read()[1]
	time.sleep(1)
	c,warped, output = bird_view(image)
	if c == 0 :
		continue

	thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY_INV,11,2)

	kernel = np.ones((15,15),np.uint8)
	bg = cv2.dilate(thresh,kernel, iterations=3)

	#cv2.imshow("intial",thresh)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	#cv2.imshow("afteropne",thresh)
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel2)
	thresh = cv2.erode(thresh,kernel,iterations = 1)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	digitCnts = []


	thresh2 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY_INV,11,2)
	# cv2.imshow("intial1",thresh2)
	kernel = np.ones((3,3),np.uint8)
	thresh2 = cv2.erode(thresh2,kernel,iterations = 1)
	#cv2.imshow("intial",thresh2)
	kernel = np.ones((2,2),np.uint8)
	thresh2 = cv2.dilate(thresh2,kernel,iterations = 2)
	thresh2 = cv2.erode(thresh2,kernel,iterations = 1)
	thresh2 = cv2.dilate(thresh2,kernel,iterations = 1)


	# thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
	# kernel = np.ones((1,2),np.uint8)
	# thresh2 = cv2.dilate(thresh2,kernel,iterations = 2)
	# kernel = np.ones((2,1),np.uint8)
	# thresh2 = cv2.dilate(thresh2,kernel,iterations = 1)






	count = 0
	# loop over the digit area candidates
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)
		# if the contour is sufficiently large, it must be a digit
		#cv2.rectangle(output,(x,y),(x+w,y+h),(0,0,255),1)
		if (w >= 7 and w<=25) and (h >= 30 and h <= 50):
			digitCnts.append(c)
			count += 1
			print(w,"   ",h)
	#cv2.imshow("edged3",output)
	#cv2.imshow("edged2",thresh)
	if count == 0  or count >= 8:
		continue

	digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]

	digits=[]
	for c in digitCnts:
		# extract the digit ROI
		(x, y, w, h) = cv2.boundingRect(c)
		if w < 8 and (h >15 and h < 30):
			x = x - 8
			w = w + 8
		roi = thresh2[y:y + h, x:x + w]
		(roiH, roiW) = roi.shape
		(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
		dHC = int(roiH * 0.1)
		V = int(roiH * 0.045)

		segments = [
    		((0+dW, 0), (w-dW, 3*dH//2)),  # top
	        ((0, 0+dH), (5*dW//3, h//2-dH)),  # top-left
	        ((w -(5*dW//3), 0+dH), (w, h // 2-dH)),  # top-right
	        ((0+dW, (h // 2) - dHC), (w-dW, (h // 2) + dHC)),  # center
	        ((0, h // 2+dH), (5*dW//3, h-dH)),  # bottom-left
	        ((w - 4*dW//3, h // 2+dH), (w, h-dH)),  # bottom-right
	        ((dW, h - 3*dH//2), (w-dW, h))  # bottom
	    ]
		on = [0] * len(segments)
       # sort the contours from left-to-right, then initialize the
       # actual digits themselves
       # loop over the segments
		for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
			segROI = roi[yA:yB, xA:xB]
			total = cv2.countNonZero(segROI)
			area = (xB - xA) * (yB - yA)
			#print(total, "  ", area)
        
		if area != 0 and total / float(area) > 0.20:
			on[i] = 1

		cv2.rectangle(output,(x+dW,y),(x+w-dW,y+(4*dH//3)),(0,0,255),1)
		cv2.rectangle(output,(x,y+dH//2),(x+(4*dW//3),y+h//2-dH//2),(0,0,255),1)
		cv2.rectangle(output,(x+w-(4*dW//3),y+dH//2),(x+w,y+h//2-dH//2),(0,0,255),1)
		cv2.rectangle(output,(x+dW,y+h//2-dHC),(x+w-dW,y+h//2+dHC),(0,0,255),1)
		cv2.rectangle(output,(x,y+h//2+dH//2),(x+(4*dW//3),y+h-dH//2),(0,0,255),1)
		cv2.rectangle(output,(x+w-4*dW//3,y+h//2+dH//2),(x+w,y+h-dH//2),(0,0,255),1)
		cv2.rectangle(output,(x+dW,y+h-3*dH//2),(x+w-dW,y+h),(0,0,255),1)

		cv2.rectangle(bg,(x+dW,y),(x+w-dW,y+(4*dH//3)),(0,0,255),1)
		cv2.rectangle(bg,(x,y+dH),(x+(5*dW//3),y+h//2-dH),(0,0,255),1)
		cv2.rectangle(bg,(x+w-(5*dW//3),y+dH),(x+w,y+h//2-dH),(0,0,255),1)
		cv2.rectangle(bg,(x+dW+V,y+h//2-dHC),(x+w-dW-V,y+h//2+dHC),(0,0,255),1)
		cv2.rectangle(bg,(x,y+h//2+dH),(x+(4*dW//3),y+h-dH),(0,0,255),1)
		cv2.rectangle(bg,(x+w-4*dW//3,y+h//2+dH),(x+w,y+h-dH),(0,0,255),1)
		cv2.rectangle(bg,(x+dW,y+h-3*dH//2),(x+w-dW,y+h),(0,0,255),1)

		#print("  *  ")
		digit =-1
		# lookup the digit and draw it on the image
		if tuple(on) in DIGITS_LOOKUP:
			digit = DIGITS_LOOKUP[tuple(on)]
		else:
			print("not found")
		if w <= 8 and digit != 100 :
			print("change")
			digit = 1

		digits.append(digit)
	print(digits)

	cv2.imshow("edged",output)
	cv2.imshow("thresh",thresh)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

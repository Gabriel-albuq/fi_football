import cv2 
import matplotlib.pyplot as plt
import time

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# read images
img1 = cv2.imread('data/neymar1.png')  
img2 = cv2.imread('data/neymar2.png') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

resize_factor = 0.5  # Adjust this factor as needed
img1 = cv2.resize(img1, None, fx=resize_factor, fy=resize_factor)
img2 = cv2.resize(img2, None, fx=resize_factor, fy=resize_factor)

#sift
orb = cv2.ORB.create()
keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)
matches = bf.match(descriptors_1,descriptors_2)

for m in matches:
    print(m.distance)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:5], None, flags=2)

cv2.imshow('SIFT', img3)
cv2.waitKey(50000)
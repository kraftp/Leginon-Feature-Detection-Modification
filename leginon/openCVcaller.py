#!/usr/bin/env python

import threading
import numpy as np
import scipy as sp
import math
import pyami.quietscipy
from scipy import ndimage
from numpy import linalg
import cv2
import sys
import operator
import copy
s_gfilt=[0, 0, 1000, 0, 0]

#-----------------------
def yshift(k1, k2, sel_matches):
    """
    y-shift:  calculate y-shift of the tilt-pair
    """
    ysd = {}
    for m in sel_matches:
        ys=(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))
        myfilt=copy.copy(s_gfilt)
        myfilt[len(myfilt)/2]*=(1/m.distance)
        filt=sp.ndimage.filters.gaussian_filter1d(myfilt, 1)
        for i in range(len(filt)):
            if (ys+i-(len(filt)/2)) in ysd:
                ysd[ys+i-(len(filt)/2)]+=filt[i]
            else:
                ysd[ys+i-(len(filt)/2)]=filt[i]
    print "y-shift:", max(ysd, key=ysd.get),"pixels with",  max(ysd.values()), "votes"
    return max(ysd, key=ysd.get)

#-----------------------
def MatchImages(image1, image2, blur=3):
    """
    Given two images:
    (1) Find regions
    (2) Match the regions
    (3) Find the affine matrix relating the two images
    
    Inputs:
        numpy image1 array, dtype=float32
        numpy image2 array, dtype=float32
        Blur the image by blur pixels (defaults to 3)

    Output:
        3x3 Affine Matrix
    """
    image1, image2 = convertImage(image1, image2)
    if blur > 0:
        image1=cv2.GaussianBlur(image1, (blur, blur), 0)
        image2=cv2.GaussianBlur(image2, (blur, blur), 0)

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("BRIEF")
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    kp1=detector.detect(image1)
    kp2=detector.detect(image2)

    k1, d1 = descriptor.compute(image1, kp1)
    k2, d2 = descriptor.compute(image2, kp2)
    print '%d keypoints in image1, %d keypoints in image2' % (len(d1), len(d2))

    matches = matcher.match(d1, d2)
    distances = [m.distance for m in matches]
    print "%d matches" % (len(distances))

    mean_dist = sum(distances)/len(distances)
    sel_matches = [m for m in matches if m.distance < mean_dist*0.6]
    ys = yshift(k1, k2, sel_matches)
    sel_matches = [m for m in sel_matches if math.fabs(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1])-ys)<10]
    print "%d matches" % (len(sel_matches))
    
    count=0
    while len(sel_matches)<40 and count<5:
        count+=1
        sel_matches = [m for m in matches if m.distance < mean_dist*(.6+.05*count)]
        ys=yshift(k1, k2, sel_matches)
        sel_matches = [m for m in sel_matches if math.fabs(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1])-ys)<10]
        print "Try:", count, "#selected matches:", len(sel_matches)
        
    ## Calculating affine matrix
    src_pts=np.float32([k1[m.queryIdx].pt for m in sel_matches ]).reshape(-1, 1, 2)
    dst_pts=np.float32([k2[m.trainIdx].pt for m in sel_matches ]).reshape(-1, 1, 2)

    affineM=cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=True)
    if affineM==None:
        print "affine matrix could not be calculated"
        return numpy.zeros([3,3], dtype=numpy.float32)        

    M=np.eye(3, dtype=float)
    for i in range(2):
        for j in range(3):
            M[i][j]=affineM[i][j]

    if math.fabs(M[0][0]*M[1][1])>1:
        print "affine matrix impossible"
        return np.zeros([3,3], dtype=np.float32)
    
    if math.fabs(math.degrees(math.acos(M[0][0]*M[1][1]))-50.)>4:
        print "affine matrix highly inaccurate"
        return np.zeros([3,3], dtype=np.float32)

    return M

#-----------------------
def convertImage(image1, image2):
    """
    Inputs:
        numpy image1 array, dtype=float32
        numpy image2 array, dtype=float32

    Output:
        numpy image1 array, dtype=uint8
        numpy image2 array, dtype=uint8   
    """
    max1 = np.amax(image1)
    max2 = np.amax(image2)
    min1 = np.amin(image1)
    min2 = np.amin(image2)

    image1 = image1*256/(max1-min1)
    image2 = image2*256/(max2-min2)

    image1 = np.asarray(image1, dtype=np.uint8)
    image2 = np.asarray(image2, dtype=np.uint8)

    return image1, image2


## image1=cv2.imread('sim_images/test1.jpg')
## image2=cv2.imread('sim_images/test2.jpg')

## M = MatchImages(image1, image2)

## print M

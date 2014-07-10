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
def radians(degrees):
        return float(degrees) * np.pi / 180.0

#-----------------------
def degrees(radians):
        return float(radians) * 180.0 / np.pi

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
    if ysd:            
        print "y-shift:", max(ysd, key=ysd.get),"pixels with",  max(ysd.values()), "votes"
        return max(ysd, key=ysd.get)
    else:
        return None

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
    image1, image2 = convertImage(image1), convertImage(image2)
    if blur > 0:
        image1=cv2.GaussianBlur(image1, (blur, blur), 0)
        image2=cv2.GaussianBlur(image2, (blur, blur), 0)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    print h1, w1, h2, w2

    view = sp.zeros((max(h1, h2), w1 + w2), sp.uint8)
    view[:h1, :w1] = image1
    view[:h2, w1:] = image2
    cv2.imwrite('sift_orig.jpg', view)


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

    mean_dist = (sum(distances)/len(distances))
    sel_matches = [m for m in matches if m.distance < mean_dist*0.6]
    ys = yshift(k1, k2, sel_matches)
    if ys is not None:
        sel_matches = [m for m in sel_matches if math.fabs(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1])-ys)<10]
    print "%d matches" % (len(sel_matches))
    
    count=0
    while len(sel_matches)<40 and count<20:
        count+=1
        sel_matches = [m for m in matches if m.distance < mean_dist*(.6+.05*count)]
        ys=yshift(k1, k2, sel_matches)
        if ys is not None:
            sel_matches = [m for m in sel_matches if math.fabs(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1])-ys)<10]
        print "Try:", count, "#selected matches:", len(sel_matches)


    for m in sel_matches:
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(view, (int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1])) , (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)
    cv2.imwrite('sift_comparison.jpg', view)
        
    ## Calculating affine matrix
    src_pts=np.float32([k1[m.queryIdx].pt for m in sel_matches ]).reshape(-1, 1, 2)
    dst_pts=np.float32([k2[m.trainIdx].pt for m in sel_matches ]).reshape(-1, 1, 2)

    affineM=cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=True)
    if affineM==None:
        print "affine matrix could not be calculated"
        return np.zeros([3,3], dtype=np.float32)        

    M=np.eye(3, dtype=float)
    for i in range(2):
        for j in range(3):
            M[i][j]=affineM[i][j]
    print M

    hview=copy.copy(view)
    for i in xrange(h1):
        for j in xrange(w1):
            vec=np.dot(M, [j, i, 1])
            if 0<int(round(vec[0]))<w2 and 0<int(round(vec[1]))<h2:
                hview[int(round(vec[1]))][int(round(vec[0]))+w1]/=2

    cv2.imwrite('sift_projection.jpg', hview)

    if math.fabs(M[0][0]*M[1][1])>1:
        print "affine matrix impossible"
        return np.zeros([3,3], dtype=np.float32)
    
    """if math.fabs(math.degrees(math.acos(M[0][0]*M[1][1]))-50.)>4:
        print "affine matrix highly inaccurate"
        return np.zeros([3,3], dtype=np.float32)"""

    ## For compatibility
    compat=copy.copy(M)
    M[0][0]=compat[1][1]
    M[2][0]=compat[1][2]
    #M[2][0]=0
    M[1][1]=compat[0][0]
    M[2][1]=compat[0][2]
    #M[2][1]=0
    M[0][2]=0.0
    M[1][2]=0.0
    print M

    return M

#-----------------------
def convertImage(image1):
    """
    Inputs:
        numpy image1 array, dtype=float32
        numpy image2 array, dtype=float32

    Output:
        numpy image1 array, dtype=uint8
        numpy image2 array, dtype=uint8   
    """
    #print image1
    
    max1 = np.amax(image1)

    min1 = np.amin(image1)
  
    #print min1, max1

    image1 = image1 - min1
    
    max1 = np.amax(image1)

   # print max1
    
    image1 = image1/max1 * 256

    image1 = np.asarray(image1, dtype=np.uint8)

    #print image1
    
    return image1

#-----------------------
def checkOpenCVResult(self, result):
        """
        Tests whether the openCV resulting affine matrix is reasonable for tilting
    Modified from original from libCV
        """
        if abs(result[0][0]) < 0.5 or abs(result[1][1]) < 0.5:
                #max tilt angle of 60 degrees
                self.logger.warning("Bad openCV result: bad tilt in matrix: "+affineToText(result))
                print ("Bad openCV result: bad tilt in matrix: "+affineToText(result))
                return False
        elif abs(result[0][0]) > 1.3 or abs(result[1][1]) > 1.3:
                #restrict maximum allowable expansion
                self.logger.warning("Bad openCV result: image expansion: "+affineToText(result))
                print ("Bad openCV result: image expansion: "+affineToText(result))
                return False
        elif abs(result[0][1]) > 0.2588 or abs(result[1][0]) > 0.2588:
                #max rotation angle of 15 degrees
                self.logger.warning("Bad openCV result: too much rotation: "+affineToText(result))
                print ("Bad openCV result: too much rotation: "+affineToText(result))
                return False
        return True

#-----------------------
def affineToText(matrix):
        """
        Extracts useful parameters from an affine homography matrix
    Modified from original from libCV
        """
        tiltv = matrix[0,0] * matrix[1,1]
        rotv = (matrix[0,1] - matrix[1,0]) / 2.0
        if abs(tiltv) > 1:
                tilt = degrees(math.acos(1.0/tiltv))
        else:
                tilt = degrees(math.acos(tiltv))
        if tilt > 90.0:
                tilt = tilt - 180.0
        if abs(rotv) < 1:
                rot = degrees(math.asin(rotv))
        else:
                rot = 180.0
        mystr = ( "tiltang = %.2f, rotation = %.2f, shift = %.2f,%.2f" %
                (tilt, rot, matrix[2,0], matrix[2,1]) )
        return mystr

#-----------------------
def FindFeatures(image, blur=3):
    """
    Given an image find regions
    
    Inputs:
    numpy image array, dtype=float32
    Blur the image by blur pixels (default=3)
        
    Output:
    List of lists of coordinates of features
    """
    image = convertImage(image)
    if blur > 0:
        image=cv2.GaussianBlur(image, (blur, blur), 0)

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("BRIEF")

    kp1=detector.detect(image)
    k1, d1 = descriptor.compute(image, kp1)

    print '%d keypoints in image' % (len(d1))

    feature_pts = [ feature.pt for feature in k1]
    return feature_pts 




## image1=cv2.imread('sim_images/test1.jpg')
## image2=cv2.imread('sim_images/test2.jpg')

## M = MatchImages(image1, image2)

## print M

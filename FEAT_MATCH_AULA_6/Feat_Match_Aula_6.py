#!/usr/bin/env python3

import numpy
import cv2
import matplotlib.pyplot as plt

def main():
    img1 = cv2.imread('/home/stigliano/Documents/Estudos/Engenharia Mecânica/5º ano/SAVI/savi_22-23/Parte06/images/castle/1.png',cv2.IMREAD_GRAYSCALE) # queryImage
    img2 = cv2.imread('/home/stigliano/Documents/Estudos/Engenharia Mecânica/5º ano/SAVI/savi_22-23/Parte06/images/castle/2.png',cv2.IMREAD_GRAYSCALE) # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

if __name__ == "__main__":
    main()

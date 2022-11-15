#!/usr/bin/env python3

import numpy
import cv2
import matplotlib


def main():

    img = cv2.imread('/home/stigliano/Documents/Estudos/Engenharia Mecânica/5º ano/SAVI/savi_22-23/Parte06/images/santorini/1.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg',img)


if __name__ == "__main__":
    main()

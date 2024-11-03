import cv2
import numpy as np


def ratio_test(desc1, desc2, r_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    m = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for x, y in m:
        if x.distance < r_thresh * y.distance:
            good.append(x)

    return good


img = cv2.imread('img.jpg')
img2 = cv2.imread('img2.jpg')

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(img2, None)

gm = ratio_test(des1, des2)

match_img = cv2.drawMatches(img, kp1, img2, kp2, gm, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

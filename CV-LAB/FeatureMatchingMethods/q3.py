import cv2
import numpy as np


def find_homography(kp1, kp2, gm):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in gm]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in gm]).reshape(-1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H, mask


def apply_ratio_test(desc1, desc2, r_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    m = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for x, y in m:
        if x.distance < r_thresh * y.distance:
            good.append(x)
    return good


img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img2.jpg')

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

gm = apply_ratio_test(des1, des2)

H, mask = find_homography(kp1, kp2, gm)

inliers = mask.ravel().tolist()
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, gm, None, matchesMask=inliers, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matches with Inliers', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Homography Matrix:\n", H)

import cv2
import numpy as np


def apply_ratio_test(desc1, desc2, r_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    m = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for x, y in m:
        if x.distance < r_thresh * y.distance:
            good.append(x)
    return good


def find_homography(kp1, kp2, good_matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H, mask


def stitch_images(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    good_matches = apply_ratio_test(des1, des2)
    H, mask = find_homography(kp1, kp2, good_matches)

    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result


img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img2.jpg')

st_img = stitch_images(img1, img2)

cv2.imshow('Stitched Image', st_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

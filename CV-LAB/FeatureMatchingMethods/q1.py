import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_sift(img_path):
    img = cv2.imread(img_path)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def compute_distances(desc1, desc2):
    dists = np.zeros((desc1.shape[0], desc2.shape[0]))
    for i in range(desc1.shape[0]):
        for j in range(desc2.shape[0]):
            diff = desc1[i] - desc2[j]
            dists[i, j] = np.sqrt(np.sum(diff ** 2))
    return dists


def find_neighbors(dists):
    return np.argmin(dists, axis=1)


def draw_matches(img1, kp1, img2, kp2, indices):
    matches = []
    for i, idx in enumerate(indices):
        if idx < len(kp2):
            matches.append(cv2.DMatch(i, idx, 0))
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


kp1, des1 = extract_sift('img.jpg')
kp2, des2 = extract_sift('img2.jpg')

dists = compute_distances(des1, des2)
inds = find_neighbors(dists)

img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img2.jpg')

img_matches = draw_matches(img1, kp1, img2, kp2, inds)

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('SIFT Matches')
plt.axis('off')
plt.show()

import cv2
import numpy as np


def scale(x):
    return (x - x.min()) / (x.max() - x.min()) * 255


def nms(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = G[i, j - 1]
                q = G[i, j + 1]

            elif 22.5 <= angle[i, j] < 67.5:
                r = G[i - 1, j + 1]
                q = G[i + 1, j - 1]

            elif 67.5 <= angle[i, j] < 112.5:
                r = G[i - 1, j]
                q = G[i + 1, j]

            elif 112.5 <= angle[i, j] < 157.5:
                r = G[i + 1, j + 1]
                q = G[i - 1, j - 1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = 25
    strong = 255

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def hysteresis(img, weak=25, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

img = cv2.imread('./img.jpeg', 0)/255
m = 5
n = 5

gauss_kernel = np.zeros((m,n), np.float32)
sigma = 1.0
for x in range(-m//2,m//2):
    for y in range(-n//2,n//2):
        normal = 1 / (2.0 * np.pi * sigma**2.0)
        exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
        gauss_kernel[y+m//2, x+n//2] = normal*exp_term

img = cv2.filter2D(src=img, ddepth=-1, kernel=gauss_kernel)

kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

grad1 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel1)
grad2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

G = scale(np.hypot(grad1, grad2))
theta = np.arctan2(grad2, grad1)

img = nms(G, theta)
img = threshold(img)
img = hysteresis(img)
img = img.astype(np.uint8)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

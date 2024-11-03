import cv2
import numpy as np

img = cv2.imread("img.jpeg", cv2.IMREAD_GRAYSCALE)

hist = np.zeros(256, dtype=np.int32)
for pixel_value in img.flatten():
    hist[pixel_value] += 1

cdf = np.cumsum(hist)

cdf_min = cdf[cdf > 0].min()  # Avoid division by zero
cdf_max = cdf.max()
norm = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
norm = np.round(norm).astype(np.uint8)

eq = norm[img]

res = np.hstack((img, eq))

cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('img.jpeg')

cv2.imshow('Original Image', img)

dw = 300
dh = 200
dp = (dw, dh)
down = cv2.resize(img, dp, interpolation=cv2.INTER_LINEAR)

uw = 600
uh = 400
upt = (uw, uh)
up = cv2.resize(img, upt, interpolation=cv2.INTER_LINEAR)

y1, y2 = 80, 280
x1, x2 = 150, 330
crop = img[y1:y2, x1:x2]

cv2.imshow('Resized Down', down)
cv2.imshow('Resized Up', up)
cv2.imshow('Cropped Image', crop)

cv2.imwrite("Cropped_Image.jpg", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()

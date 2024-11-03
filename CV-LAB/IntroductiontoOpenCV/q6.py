import cv2
img = cv2.imread("flower.jpg")
rot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Ulta_flower',rot)
cv2.waitKey(0)
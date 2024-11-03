import cv2
img = cv2.imread('flower.jpg')
cv2.imshow('Picture',img)
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_flower.jpg',gray_image)
cv2.waitKey(0)
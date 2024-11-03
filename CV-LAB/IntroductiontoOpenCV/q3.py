import cv2
img = cv2.imread("flower.jpg")
color = img[100,100]
print(color)
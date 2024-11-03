import cv2
img = cv2.imread("flower.jpg")
width = 300
height = 200

res = cv2.resize(img,(width,height))
cv2.imwrite('Small_flower.jpg',res)
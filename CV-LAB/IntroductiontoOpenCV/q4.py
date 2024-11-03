import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
# Create a blank image of size 512x512 with 3 color channels (RGB)
cv2.rectangle(img,(260,0),(400,128),(0,0,255),5)
# top left, bottom right, colour, thickness
cv2.imshow('Rectangle',img)
cv2.waitKey(0)

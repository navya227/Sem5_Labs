import cv2
import numpy as np

def kmeans(ip, k=8, op='kmeans_output.jpg'):
    img = cv2.imread(ip)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    cri = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, label, cen = cv2.kmeans(Z, k, None, cri, 10, cv2.KMEANS_RANDOM_CENTERS)
    cen = np.uint8(cen)
    new = cen[label.flatten()]
    new = new.reshape(img.shape)
    cv2.imwrite(op, new)
    cv2.imshow('Original Image', img)
    cv2.imshow('K-means Clustered Image', new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
ip = 'img.jpg'
op = 'kmeans_output.jpg'
kmeans(ip, k=8, op=op) 

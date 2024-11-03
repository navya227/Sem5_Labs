import cv2
import numpy as np
import matplotlib.pyplot as plt


def disp(title, img, cmap='gray'):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


def seg_col(img_path, lb, ub):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lb = np.array(lb)
    ub = np.array(ub)
    mask = cv2.inRange(hsv_img, lb, ub)
    seg = cv2.bitwise_and(img, img, mask=mask)
    disp('Original Image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    disp('Mask', mask, cmap='gray')
    disp('Segmented Image', cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))


path = 'img.jpg'
lb = [150, 100, 100]
ub = [200, 255, 255]
seg_col(path, lb, ub)

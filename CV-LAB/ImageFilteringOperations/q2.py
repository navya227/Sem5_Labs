import cv2
import numpy as np


def convo(img, ker):

    ker_h, ker_w = ker.shape
    pad_h = ker_h // 2
    pad_w = ker_w // 2

    pad_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            result[y, x] = np.sum(ker * pad_img[y:y + ker_h, x:x + ker_w])

    return result


def grad(img):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = convo(img, sobel_x)
    grad_y = convo(img, sobel_y)

    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    return grad_x, grad_y, mag


def main():
    img = cv2.imread('img.jpeg')
    grad_x, grad_y, mag = grad(img)
    cv2.imwrite('grad_x.jpg', grad_x)
    cv2.imwrite('grad_y.jpg', grad_y)
    cv2.imwrite('mag.jpg', mag)

if __name__ == "__main__":
    main()

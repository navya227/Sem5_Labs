import cv2
import numpy as np

def apply_ker(image, ker):
    rows, cols = image.shape
    ksize = ker.shape[0]
    pad = ksize // 2

    pad_img = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    op = np.zeros_like(image, dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            reg = pad_img[i:i + ksize, j:j + ksize]
            op[i, j] = np.sum(reg * ker)

    return op

def box_fil(image, ker_size):
    box_ker = np.ones((ker_size, ker_size), dtype=np.float64) / (ker_size * ker_size)
    return apply_ker(image, box_ker)

def gaussian_fil(image, ker_size, sigma):
    ax = np.arange(-ker_size // 2 + 1., ker_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    ker /= np.sum(ker)
    return apply_ker(image, ker)

def main():
    image = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)

    ker_size = 5
    sigma = 1.0

    box_img = box_fil(image, ker_size)
    gaus_img = gaussian_fil(image, ker_size, sigma)

    box_img = cv2.normalize(box_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gaus_img = cv2.normalize(gaus_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow('Original Image', image)
    cv2.imshow('Box Filter Output', box_img)
    cv2.imshow('Gaussian Filter Output', gaus_img)

    cv2.imwrite('box_img.jpg', box_img)
    cv2.imwrite('gaus_img.jpg', gaus_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


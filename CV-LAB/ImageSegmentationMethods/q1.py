import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(ip):
    img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    return img

def glob(img, tval):
    bin = np.where(img >= tval, 255, 0).astype(np.uint8)
    return bin

def adap(img, bs, C):
    bin = np.zeros_like(img, dtype=np.uint8)
    rows, cols = img.shape

    pimg = np.pad(img, pad_width=((bs//2, bs//2), (bs//2, bs//2)), mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            blo = pimg[i:i+bs, j:j+bs]
            mn = np.mean(blo)
            thresh = mn - C
            bin[i, j] = 255 if img[i, j] >= thresh else 0

    return bin

def disp(imgs, titles):
    plt.figure(figsize=(12, 8))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ip = 'img.jpg'

    img = load_img(ip)

    gt = glob(img, tval=127)
    at = adap(img, bs=11, C=2)

    imgs = [img, gt, at]
    titles = ['Original Image', 'Global Thresholding', 'Adaptive Thresholding']
    disp(imgs, titles)

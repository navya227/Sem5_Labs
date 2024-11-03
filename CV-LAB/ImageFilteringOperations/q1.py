import cv2
import numpy as np

def custom(image, blur):
    kernel = np.ones((blur, blur), np.float32) / (blur * blur)
    return cv2.filter2D(image, -1, kernel)

def unsharp(image, blur, alpha):
    blur1 = custom(image, blur)
    img2 = image.astype(np.float32)
    blur2 = blur1.astype(np.float32)
    mask = cv2.subtract(img2, blur2)
    sharp2 = cv2.add(img2, alpha * mask)
    sharp = np.clip(sharp2, 0, 255).astype(np.uint8)
    return sharp

def main():
    image = cv2.imread('img.jpeg')
    blur = 5
    alpha = 1.5
    sharp = unsharp(image, blur, alpha)
    cv2.imshow('Original Image', image)
    cv2.imshow('Sharpened Image', sharp)
    cv2.imwrite('sharp.jpg', sharp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



import cv2
import numpy as np
import matplotlib.pyplot as plt

def fast(img_path, threshold=20):
    img = cv2.imread(img_path)
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gr = np.uint8(gr)

    ct = threshold

    c = np.zeros_like(gr)

    pix = [(int(3 * np.cos(2 * np.pi * i / 16)), int(3 * np.sin(2 * np.pi * i / 16))) for i in range(16)]

    def is_corner(x, y):
        if x < 3 or x >= gr.shape[0] - 3 or y < 3 or y >= gr.shape[1] - 3:
            return False

        pixel_value = gr[x, y]
        count = 0

        for dx, dy in pix:
            if abs(gr[x + dx, y + dy] - pixel_value) > ct:
                count += 1

        return count >= 12

    for x in range(gr.shape[0]):
        for y in range(gr.shape[1]):
            if is_corner(x, y):
                c[x, y] = 255

    img[c == 255] = [0, 0, 255]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('FAST Corner Detection')
    plt.axis('off')
    plt.show()

fast('Untitled.jpeg')

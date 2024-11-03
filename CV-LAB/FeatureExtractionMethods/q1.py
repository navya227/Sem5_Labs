import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris(image_path, k=0.04, th=1e-6):
    image = cv2.imread(image_path)
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gr = np.float32(gr)

    Ix = cv2.Sobel(gr, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gr, cv2.CV_64F, 0, 1, ksize=3)

    Ix2 = Ix**2
    Iy2 = Iy**2
    IxIy = Ix * Iy

    window_size = 3
    o = window_size // 2
    res = np.zeros_like(gr)

    for i in range(o, gr.shape[0] - o):
        for j in range(o, gr.shape[1] - o):
            Sx2 = np.sum(Ix2[i-o:i+o+1, j-o:j+o+1])
            Sy2 = np.sum(Iy2[i-o:i+o+1, j-o:j+o+1])
            Sxy = np.sum(IxIy[i-o:i+o+1, j-o:j+o+1])

            M = np.array([[Sx2, Sxy], [Sxy, Sy2]])
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            res[i, j] = det_M - k * (trace_M ** 2)

    corners = res > th * res.max()
    image[corners] = [0, 0, 255]

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection')
    plt.axis('off')
    plt.show()

harris('Untitled.jpeg')

import cv2
import numpy as np
import matplotlib.pyplot as plt


def disp(title, img, cmap='gray'):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


def hough(ed, rho_res=1, theta_res=np.pi / 180, threshold=100):
    h, w = ed.shape
    diag = int(np.sqrt(w**2 + h**2))
    num_rhos = int(2 * diag / rho_res)
    num_thetas = int(np.pi / theta_res)
    acc = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    thetas = np.arange(0, np.pi, theta_res)
    yc, xc = np.nonzero(ed)

    for x, y in zip(xc, yc):
        for theta in thetas:
            rho = x * np.cos(theta) + y * np.sin(theta)
            ri = int(rho + diag) // rho_res
            ti = int(theta / theta_res) % num_thetas
            acc[ri, ti] += 1

    l = []
    for rho in range(num_rhos):
        for theta in range(num_thetas):
            if acc[rho, theta] > threshold:
                rval = (rho - diag) * rho_res
                tval = theta * theta_res
                l.append((rval, tval))

    return acc, l


def draw(img, l):
    col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape

    for rho, theta in l:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + h * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - h * a)
        cv2.line(col, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return col


if __name__ == "__main__":
    path = 'grid.png'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ed = cv2.Canny(blur, 50, 150)
    acc, l = hough(ed, threshold=100)
    lin_img = draw(image, l)
    disp('Original Image', image)
    disp('Edges', ed)
    disp('Accumulator', np.log(acc + 1))
    disp('Detected Lines', lin_img)

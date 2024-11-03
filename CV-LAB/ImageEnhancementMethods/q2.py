import cv2
import numpy as np


def hist(img):
    pix = img.flatten()

    hist = np.zeros(256, dtype=np.float64)

    for value in pix:
        hist[int(value)] += 1

    cdf = np.cumsum(hist)
    norm = cdf / cdf[-1]  # Normalize CDF to range [0, 1]

    return hist, norm


def hist_spec(in_path, ref_path, out_path):
    inp = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    _, cdf_in = hist(inp)
    _, cdf_ref = hist(ref)

    m = np.interp(cdf_in, cdf_ref, np.arange(256))

    spec_img = np.interp(inp, np.arange(256), m).astype(np.uint8)

    cv2.imwrite(out_path, spec_img)
    cv2.imshow('Specified Image', spec_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


hist_spec('img.jpeg', 'dog.jpg', 'new.jpg')

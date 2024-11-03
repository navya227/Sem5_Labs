import cv2
import numpy as np


def apply_kernel(image, kernel):
    rows, cols = image.shape
    ksize = kernel.shape[0]
    pad = ksize // 2

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + ksize, j:j + ksize]
            output[i, j] = np.sum(region * kernel)

    return output


def laplacian_edge_detection(image):
    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float64)

    laplacian_edges = apply_kernel(image, laplacian_kernel)

    # Normalize the result to the range [0, 255]
    laplacian_edges = np.abs(laplacian_edges)
    normalized_edges = cv2.normalize(laplacian_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_edges


def main():
    image_path = 'image.jpg'  # Change this to your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    edge_magnitude = laplacian_edge_detection(image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Detection', edge_magnitude)

    cv2.imwrite('laplacian_edge_detection.jpg', edge_magnitude)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

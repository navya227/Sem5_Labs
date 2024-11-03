import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def compute_gradients(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    return magnitude, angle

def compute_histograms(magnitude, angle, cell_size=8, num_bins=9):
    rows, cols = magnitude.shape
    cell_rows = rows // cell_size
    cell_cols = cols // cell_size
    histograms = np.zeros((cell_rows, cell_cols, num_bins))

    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            hist, _ = np.histogram(cell_ang, bins=num_bins, range=(0, 180), weights=cell_mag)
            histograms[i, j, :] = hist

    return histograms

def normalize_blocks(histograms, block_size=2, eps=1e-5):
    cell_rows, cell_cols, num_bins = histograms.shape
    block_rows = cell_rows - block_size + 1
    block_cols = cell_cols - block_size + 1
    normalized_blocks = np.zeros((block_rows, block_cols, block_size * block_size * num_bins))

    for i in range(block_rows):
        for j in range(block_cols):
            block_histogram = histograms[i:i+block_size, j:j+block_size, :].flatten()
            norm = np.sqrt(np.sum(block_histogram**2) + eps**2)
            normalized_blocks[i, j, :] = block_histogram / norm

    return normalized_blocks

def extract_hog_features(image, window_size, cell_size, block_size):
    magnitude, angle = compute_gradients(image)
    histograms = compute_histograms(magnitude, angle, cell_size)
    blocks = normalize_blocks(histograms, block_size)

    window_hogs = []
    win_rows, win_cols = window_size[1] // cell_size, window_size[0] // cell_size
    for i in range(0, blocks.shape[0] - win_rows + 1):
        for j in range(0, blocks.shape[1] - win_cols + 1):
            hog_feature = blocks[i:i+win_rows, j:j+win_cols, :].flatten()
            window_hogs.append((i * cell_size, j * cell_size, hog_feature))

    return window_hogs

def calculate_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[idxs[:last]])

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]

def draw_boxes(image, boxes):
    for (x, y, w, h, _) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main():
    image_path = 'human.jpeg'
    reference_path = 'reference.jpeg'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    # Compute HoG features for reference image
    ref_magnitude, ref_angle = compute_gradients(reference_image)
    ref_histograms = compute_histograms(ref_magnitude, ref_angle)
    ref_blocks = normalize_blocks(ref_histograms)
    ref_features = ref_blocks.flatten()

    # Detect objects in the test image
    window_size = (64, 128)
    cell_size = 8
    block_size = 2
    threshold = 0.5

    detections = extract_hog_features(image, window_size, cell_size, block_size)
    similar_windows = []

    for (x, y, features) in detections:
        if len(features) == len(ref_features):
            similarity = calculate_similarity(ref_features, features)
            if similarity > threshold:
                similar_windows.append((x, y, window_size[0], window_size[1], similarity))
        else:
            print(f"Feature size mismatch: ref_features {len(ref_features)}, window feature {len(features)}")

    # Apply non-max suppression
    filtered_boxes = non_max_suppression(similar_windows, 0.3)

    # Draw and show the results
    image_with_boxes = draw_boxes(image, filtered_boxes)
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title('Detected Humans')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

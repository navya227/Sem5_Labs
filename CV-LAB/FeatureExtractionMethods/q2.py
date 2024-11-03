import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_gaussian_pyramid(img, num_scales, k=1.2):
    """Generate a Gaussian pyramid."""
    pyramid = [img]
    sigma = 1.0
    for _ in range(num_scales - 1):
        sigma *= k
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        pyramid.append(blurred)
    return pyramid

def build_dog_pyramid(gaussian_pyramid):
    """Generate a Difference of Gaussians (DoG) pyramid."""
    dog_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        dog = cv2.subtract(gaussian_pyramid[i + 1], gaussian_pyramid[i])
        dog_pyramid.append(dog)
    return dog_pyramid

def detect_keypoints(dog_pyramid, threshold=0.03):
    """Detect keypoints in the DoG images."""
    keypoints = []
    for i, dog in enumerate(dog_pyramid):
        # Find local extrema
        max_filter = np.maximum.reduce([
            np.roll(dog, shift, axis=axis) for shift in (-1, 1) for axis in (0, 1)
        ])
        min_filter = np.minimum.reduce([
            np.roll(dog, shift, axis=axis) for shift in (-1, 1) for axis in (0, 1)
        ])
        local_max = (dog == max_filter)
        local_min = (dog == min_filter)
        mask = local_max | local_min
        mask = np.abs(dog) > threshold * np.max(np.abs(dog))
        keypoints.extend([(i, x, y) for x, y in zip(*np.nonzero(mask))])
    return keypoints

def compute_descriptors(img, keypoints, patch_size=16):
    """Compute descriptors for each keypoint."""
    descriptors = []
    half_patch = patch_size // 2

    for scale_idx, x, y in keypoints:
        scale = 1.2 ** scale_idx
        x, y = int(x), int(y)
        patch = img[max(0, x - half_patch):x + half_patch,
                      max(0, y - half_patch):y + half_patch]

        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            continue

        patch = cv2.resize(patch, (patch_size, patch_size))
        patch = patch.astype(np.float32)
        desc = patch.flatten()
        desc /= np.linalg.norm(desc)
        descriptors.append(desc)

    return np.array(descriptors)

def draw_keypoints(img, keypoints):
    """Draw keypoints on the image."""
    img_kp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for scale_idx, x, y in keypoints:
        cv2.circle(img_kp, (int(y), int(x)), 4, (0, 255, 0), -1)
        cv2.putText(img_kp, f'{scale_idx}', (int(y), int(x) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return img_kp

def main():
    img_path = 'Untitled.jpeg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Build Gaussian pyramid
    gaussian_pyramid = build_gaussian_pyramid(img, num_scales=3)

    # Build DoG pyramid
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)

    # Detect keypoints
    keypoints = detect_keypoints(dog_pyramid)

    # Compute descriptors
    descriptors = compute_descriptors(img, keypoints)

    # Draw keypoints
    img_kp = draw_keypoints(img, keypoints)

    # Display results
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title('Detected Keypoints')
    plt.axis('off')
    plt.show()

    print(f"Descriptors shape: {descriptors.shape}")

if __name__ == "__main__":
    main()

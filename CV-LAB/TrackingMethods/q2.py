import cv2
import numpy as np

def harris_corner_detection(frame):
    """Detects corners using the Harris corner detector."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # Detect corners
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)  # Dilate to mark the corners

    # Extract corner positions
    threshold = 0.01 * corners.max()
    keypoints = np.argwhere(corners > threshold)  # Get points above the threshold
    keypoints = np.flip(keypoints, axis=1)  # Convert (y, x) to (x, y)

    return np.float32(keypoints).reshape(-1, 1, 2)

def klt_tracker(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    # Detect initial keypoints using Harris corner detector
    initial_pts = harris_corner_detection(first_frame)

    # Define LK optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = initial_pts

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (KLT)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        # Select good points
        good_new = curr_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Visualize the tracked points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            frame = cv2.arrowedLine(frame, (int(x_old), int(y_old)), (int(x_new), int(y_new)),
                                    (0, 255, 0), 2, tipLength=0.4)

        # Show the tracking result
        cv2.imshow('KLT Tracker', frame)

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the KLT Tracker on a video
video_path = "test.mp4"  # Replace with your video path
klt_tracker(video_path)

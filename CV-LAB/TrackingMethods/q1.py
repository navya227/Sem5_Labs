import cv2
import numpy as np

def lucas_kanade_motion_visualization(video_path):
    # Initialize SIFT feature detector
    sift = cv2.SIFT_create()
    
    # Read video input
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(prev_gray, None)

    # Optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray, None)

        # FLANN-based feature matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match features between frames
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Extract coordinates of matched keypoints
        prev_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Visualize the motion between frames by drawing arrows
        motion_frame = frame.copy()
        for i, (p0, p1) in enumerate(zip(prev_pts, curr_pts)):
            x0, y0 = p0.ravel()
            x1, y1 = p1.ravel()
            # Draw the motion arrow
            motion_frame = cv2.arrowedLine(motion_frame, (int(x0), int(y0)), (int(x1), int(y1)),
                                           (0, 255, 0), 2, tipLength=0.4)

        # Show the motion visualization
        cv2.imshow('Motion Visualization', motion_frame)

        # Update previous frame and keypoints for the next iteration
        prev_frame = frame.copy()
        kp1, des1 = kp2, des2

        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the tracker on a sample video
video_path = "test.mp4"  # Provide path to your video
lucas_kanade_motion_visualization(video_path)

import cv2
vid = cv2.VideoCapture('video.mp4')
while True:
    ret,frame = vid.read()
    if not ret:
        break
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

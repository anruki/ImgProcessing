import cv2
import math
import numpy as np


center_points_prev_frame = []
tracking_objects = {}
track_id = 0
count = 0

cap = cv2.VideoCapture("blueangels.mp4")

# Comprobar que el vídeo se abre correctamente
if not cap.isOpened():
    print("Error opening video file")
    exit()
else:
    print("success")

# Create background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    center_points_cur_frame = []
    count += 1

    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply Gaussian blur
    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_gray = cv2.GaussianBlur(f_gray, (15, 15), 0)

    height, width = f_gray.shape

    # Threshold to create a mask for dark regions
    _, dark_mask = cv2.threshold(f_gray, 100, 255, cv2.THRESH_BINARY_INV)


    # Find contours in the combined mask
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate area and filter small objects
        area = cv2.contourArea(cnt)
        if 1700 < area :
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = int(x + w / 2), int(y + h / 2)
            center_points_cur_frame.append((cx, cy))

            # Draw the rectangle on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Update object tracking
    if count <= 2:
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1
    else:
        updated_objects = {}
        for pt in center_points_cur_frame:
            match_found = False
            for object_id, prev_pt in tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if distance < 20:
                    updated_objects[object_id] = pt
                    match_found = True
                    break

            if not match_found:
                tracking_objects[track_id] = pt
                updated_objects[track_id] = pt
                track_id += 1

        tracking_objects = updated_objects

    # Draw circles and IDs
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(
            frame, f" {object_id}", (pt[0], pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Dark Mask", dark_mask)

    # Copy current frame points to previous frame
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()

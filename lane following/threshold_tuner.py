import cv2
import glob
import numpy as np

# --------------------------------------------------
# IMAGE SETTINGS
# --------------------------------------------------

IMAGE_FOLDER = "lane_images"

images = sorted(glob.glob(f"{IMAGE_FOLDER}/*.jpg"))

if not images:
    raise SystemExit("No JPG images found in images/raw/")

image_index = 0

# --------------------------------------------------
# TRACKBAR CALLBACK
# --------------------------------------------------

def nothing(x):
    pass


# --------------------------------------------------
# WINDOWS
# --------------------------------------------------

cv2.namedWindow("Camera")
cv2.namedWindow("Mask")
cv2.namedWindow("Trackbars")

cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)

cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S_max", "Trackbars", 100, 255, nothing)

cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", 50, 255, nothing)

print("Controls:")
print("N = next image")
print("P = previous image")
print("Q = quit and print final values")

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    frame = cv2.imread(images[image_index])

    if frame is None:
        print("Could not read:", images[image_index])
        image_index = (image_index + 1) % len(images)
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")

    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")

    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    cv2.putText(
        frame,
        f"Image: {image_index + 1}/{len(images)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        image_index = (image_index + 1) % len(images)

    elif key == ord("p"):
        image_index = (image_index - 1) % len(images)

    elif key == ord("q"):

        print("\nFinal threshold values:")
        print(f"LOWER = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"UPPER = np.array([{h_max}, {s_max}, {v_max}])")

        break

cv2.destroyAllWindows()
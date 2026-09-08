import cv2
import glob
import numpy as np
import pandas as pd
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_FOLDER = "images/raw"

OUTPUT_FOLDER = "output/detected"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------
# YOUR TUNED HSV VALUES
# --------------------------------------------------

LOWER = np.array([0, 0, 0])
UPPER = np.array([179, 100, 50])

# --------------------------------------------------
# PIXEL TO CM
# --------------------------------------------------

with open("calibration/pixel_cm_ratio.txt") as f:
    CM_PER_PIXEL = float(f.read().strip())

# --------------------------------------------------
# ROI
# --------------------------------------------------

ROI_TOP = 0.65

# --------------------------------------------------
# MORPHOLOGICAL CLEANUP
# --------------------------------------------------

KERNEL = np.ones((5, 5), np.uint8)

# --------------------------------------------------
# IMAGES
# --------------------------------------------------

images = sorted(glob.glob(f"{IMAGE_FOLDER}/*.jpg"))

if not images:
    raise SystemExit("No images found")

results = []

# --------------------------------------------------
# PROCESS EACH IMAGE
# --------------------------------------------------

for image_path in images:

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    h, w = frame.shape[:2]

    frame_center_x = w // 2

    roi_top = int(h * ROI_TOP)

    roi = frame[roi_top:h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER, UPPER)

    # Close small gaps caused by glare
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    # Remove tiny noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    display = frame.copy()

    cv2.line(
        display,
        (frame_center_x, 0),
        (frame_center_x, h),
        (255, 0, 0),
        2
    )

    cv2.line(
        display,
        (0, roi_top),
        (w, roi_top),
        (0, 255, 255),
        1
    )

    offset_px = None
    offset_cm = None
    direction = "NOT_DETECTED"

    if contours:

        largest = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest)

        if area > 80:

            M = cv2.moments(largest)

            if M["m00"] != 0:

                cx = int(M["m10"] / M["m00"])

                cy = int(M["m01"] / M["m00"]) + roi_top

                offset_px = cx - frame_center_x

                offset_cm = offset_px * CM_PER_PIXEL

                if offset_cm > 0:
                    direction = "RIGHT"
                elif offset_cm < 0:
                    direction = "LEFT"
                else:
                    direction = "CENTER"

                cv2.circle(
                    display,
                    (cx, cy),
                    8,
                    (0, 255, 0),
                    -1
                )

                text = f"Offset: {abs(offset_cm):.2f} cm {direction}"

                cv2.putText(
                    display,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                text = "Invalid contour"

        else:

            text = "Line too small/noisy"

    else:

        text = "Line NOT detected"

    cv2.putText(
        display,
        text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    cv2.imwrite(output_path, display)

    mask_name = os.path.splitext(filename)[0] + "_mask.jpg"

    cv2.imwrite(
        os.path.join("output/masks", mask_name),
        mask
    )

    results.append({
        "image": filename,
        "offset_px": offset_px,
        "offset_cm": offset_cm,
        "direction": direction
    })

    print(filename, "=>", text)

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    "output/measurements.csv",
    index=False
)

print("\nFinished.")
print("Detected images saved in output/detected/")
print("Masks saved in output/masks/")
print("Measurements saved in output/measurements.csv")
import cv2
import glob
import numpy as np
import pandas as pd
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_FOLDER = "images/raw"

OUTPUT_FOLDER = "output/coefficients"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------
# HSV VALUES
# --------------------------------------------------

LOWER = np.array([0, 0, 0])
UPPER = np.array([179, 100, 50])

# --------------------------------------------------
# ROI
# --------------------------------------------------

ROI_TOP = 0.65

KERNEL = np.ones((5, 5), np.uint8)

images = sorted(glob.glob(f"{IMAGE_FOLDER}/*.jpg"))

results = []

for image_path in images:

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    h, w = frame.shape[:2]

    roi_top = int(h * ROI_TOP)

    roi = frame[roi_top:h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER, UPPER)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    # --------------------------------------------------
    # GET WHITE PIXELS
    # --------------------------------------------------

    ys, xs = np.where(mask > 0)

    if len(xs) < 20:

        print(image_path, "=> Not enough points")

        continue

    # Convert ROI y coordinates to full image coordinates
    ys = ys + roi_top

    # --------------------------------------------------
    # POLYNOMIAL FIT
    # --------------------------------------------------

    coefficients = np.polyfit(
        ys,
        xs,
        2
    )

    a, b, c = coefficients

    # --------------------------------------------------
    # DRAW CURVE
    # --------------------------------------------------

    display = frame.copy()

    y_values = np.linspace(
        roi_top,
        h - 1,
        100
    )

    x_values = a * y_values**2 + b * y_values + c

    for x, y in zip(x_values, y_values):

        x = int(x)
        y = int(y)

        if 0 <= x < w and 0 <= y < h:

            cv2.circle(
                display,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    text = f"x = {a:.6e}y^2 + {b:.6e}y + {c:.3f}"

    cv2.putText(
        display,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    filename = os.path.basename(image_path)

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, filename),
        display
    )

    results.append({
        "image": filename,
        "a": a,
        "b": b,
        "c": c
    })

    print(filename)
    print(f"a = {a:.8e}")
    print(f"b = {b:.8e}")
    print(f"c = {c:.5f}")
    print()

# --------------------------------------------------
# SAVE COEFFICIENTS
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    "output/coefficients/lane_coefficients.csv",
    index=False
)

print("Saved lane coefficients.")
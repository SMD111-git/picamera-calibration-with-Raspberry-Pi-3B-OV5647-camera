import cv2
import math

IMAGE_PATH = "images/raw/lane_001.jpg"

REAL_DISTANCE_CM = float(
    input("Enter the real distance between the two points in cm: ")
)

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise SystemExit("Could not read image")

points = []


def click_event(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 2:

            points.append((x, y))

            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

            cv2.imshow("Calibration", frame)

            print(f"Point {len(points)}: ({x}, {y})")


cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", click_event)

print("Click two points that are a known distance apart.")
print("Press any key after selecting both points.")

cv2.imshow("Calibration", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 2:
    raise SystemExit("Exactly two points are required.")

(x1, y1), (x2, y2) = points

pixel_distance = math.sqrt(
    (x2 - x1) ** 2 +
    (y2 - y1) ** 2
)

cm_per_pixel = REAL_DISTANCE_CM / pixel_distance

print("\nCalibration result:")
print(f"Pixel distance: {pixel_distance:.2f}")
print(f"Real distance: {REAL_DISTANCE_CM:.2f} cm")
print(f"cm per pixel: {cm_per_pixel:.6f}")

with open("calibration/pixel_cm_ratio.txt", "w") as f:
    f.write(str(cm_per_pixel))

print("\nSaved to calibration/pixel_cm_ratio.txt")
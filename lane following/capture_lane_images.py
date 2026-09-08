# this code on pi
# capture_lane_images.py
import cv2
from picamera2 import Picamera2
import os

SAVE_DIR = "lane_images"
os.makedirs(SAVE_DIR, exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()

print("Press SPACE to capture, 'q' to quit.")
count = 0
while True:
    frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
    cv2.imshow("Capture - SPACE to save, q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        path = os.path.join(SAVE_DIR, f"lane_{count:03d}.jpg")
        cv2.imwrite(path, frame)
        print(f"Saved {path}")
        count += 1
    elif key == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
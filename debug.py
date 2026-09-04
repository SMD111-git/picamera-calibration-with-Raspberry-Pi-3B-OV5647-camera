import cv2
import glob
import numpy as np

# Try both orientations in case the pattern is rotated relative to what we assumed
CANDIDATE_SIZES = [(6, 8), (8, 6), (7, 9), (9, 7), (6, 9), (9, 6)]

images = sorted(glob.glob("images/*.jpg"))
if not images:
    raise SystemExit("No images found in images/ folder.")

# First, figure out which board size actually detects on ANY image
print("Testing candidate board sizes on all images...\n")
best_size = None
best_count = 0

for size in CANDIDATE_SIZES:
    count = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(
            gray, size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        if found:
            count += 1
    print(f"Board size {size}: detected in {count}/{len(images)} images")
    if count > best_count:
        best_count = count
        best_size = size

print(f"\nBest candidate: {best_size} with {best_count}/{len(images)} detections")

if best_count == 0:
    print("\nNo board size worked on any image. This usually means:")
    print("- The board isn't fully visible / is cropped in most photos")
    print("- Images are too blurry or too dark")
    print("- The printed pattern isn't the standard OpenCV chessboard")
    raise SystemExit()

# Now visually show results with the best size found, one by one
print(f"\nShowing detection results for size {best_size}. Press any key to advance, 'q' to quit.")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, best_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    vis = img.copy()
    if found:
        cv2.drawChessboardCorners(vis, best_size, corners, found)
        label = "FOUND"
    else:
        label = "NOT FOUND"
    cv2.putText(vis, f"{fname} - {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if not found else (0, 255, 0), 2)
    # resize for display if huge
    h, w = vis.shape[:2]
    if w > 1280:
        scale = 1280 / w
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    cv2.imshow("Debug", vis)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
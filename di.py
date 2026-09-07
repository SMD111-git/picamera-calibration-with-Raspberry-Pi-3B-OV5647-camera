import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

# ============================================================
# SETTINGS
# ============================================================

CHESSBOARD = (7, 9)
SQUARE_SIZE = 0.020

CALIBRATION_FILE = "calibration_data/calibration.npz"

# ============================================================
# LOAD CALIBRATION
# ============================================================

data = np.load(CALIBRATION_FILE)

mtx = data["mtx"]
dist = data["dist"]

print("Camera Matrix:")
print(mtx)

print("\nDistortion:")
print(dist.ravel())

# ============================================================
# CHESSBOARD 3D POINTS
# ============================================================

objp = np.zeros(
    (CHESSBOARD[0] * CHESSBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD[0],
    0:CHESSBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# ============================================================
# FIND CHESSBOARD POSES AGAIN
# ============================================================

images = glob.glob("images/*.jpg")

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

poses = []

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD,
        None
    )

    if not found:
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )

    # Solve camera pose
    success, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        mtx,
        dist
    )

    if success:
        poses.append((rvec, tvec, fname))


print(f"\nFound poses for {len(poses)} images.")

# ============================================================
# 3D PLOT
# ============================================================

fig = plt.figure(figsize=(12, 9))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# ============================================================
# PLOT CAMERA
# ============================================================

camera_pos = np.array([0, 0, 0])

ax.scatter(
    0,
    0,
    0,
    color="red",
    s=100,
    label="Camera"
)

# Camera coordinate axes
axis_length = 0.15

ax.quiver(
    0, 0, 0,
    axis_length, 0, 0,
    color="red",
    linewidth=3
)

ax.quiver(
    0, 0, 0,
    0, axis_length, 0,
    color="green",
    linewidth=3
)

ax.quiver(
    0, 0, 0,
    0, 0, axis_length,
    color="blue",
    linewidth=3
)

ax.text(
    axis_length,
    0,
    0,
    "X"
)

ax.text(
    0,
    axis_length,
    0,
    "Y"
)

ax.text(
    0,
    0,
    axis_length,
    "Z"
)

# ============================================================
# PLOT EACH CHESSBOARD
# ============================================================

for i, (rvec, tvec, fname) in enumerate(poses):

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Chessboard points in camera coordinates
    board_points = (
        R @ objp.T
    ).T + tvec.reshape(1, 3)

    x = board_points[:, 0]
    y = board_points[:, 1]
    z = board_points[:, 2]

    # Draw chessboard points
    ax.scatter(
        x,
        y,
        z,
        s=15,
        alpha=0.6
    )

    # Draw chessboard outline
    corners_3d = np.array([
        board_points[0],
        board_points[CHESSBOARD[0] - 1],
        board_points[-1],
        board_points[
            -CHESSBOARD[0]
        ]
    ])

    ax.plot(
        corners_3d[:, 0],
        corners_3d[:, 1],
        corners_3d[:, 2],
        color="black",
        linewidth=1
    )

    # Close outline
    ax.plot(
        [corners_3d[-1, 0], corners_3d[0, 0]],
        [corners_3d[-1, 1], corners_3d[0, 1]],
        [corners_3d[-1, 2], corners_3d[0, 2]],
        color="black"
    )

    # Draw center
    center = board_points.mean(axis=0)

    ax.text(
        center[0],
        center[1],
        center[2],
        str(i + 1),
        fontsize=8
    )

# ============================================================
# LABELS
# ============================================================

ax.set_title(
    "3D Camera Calibration Visualization",
    fontsize=16
)

ax.set_xlabel("Camera X (meters)")
ax.set_ylabel("Camera Y (meters)")
ax.set_zlabel("Camera Z (meters)")

ax.legend()

# ============================================================
# EQUAL AXIS SCALE
# ============================================================

all_points = []

for rvec, tvec, fname in poses:

    R, _ = cv2.Rodrigues(rvec)

    points = (
        R @ objp.T
    ).T + tvec.reshape(1, 3)

    all_points.append(points)

if all_points:

    all_points = np.vstack(all_points)

    max_range = (
        all_points.max(axis=0)
        - all_points.min(axis=0)
    ).max() / 2

    mid = (
        all_points.max(axis=0)
        + all_points.min(axis=0)
    ) / 2

    ax.set_xlim(
        mid[0] - max_range,
        mid[0] + max_range
    )

    ax.set_ylim(
        mid[1] - max_range,
        mid[1] + max_range
    )

    ax.set_zlim(
        mid[2] - max_range,
        mid[2] + max_range
    )

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# -------------------------
# SETTINGS
# -------------------------

# Path to checkerboard images (use WITHOUT object images ideally)
image_folder = "../dataset/FD/without_object/*.jpeg"

# Checkerboard size (number of inner corners)
CHECKERBOARD = (9,6)   # change if your grid is different

# Square size (optional, only affects scale)
square_size = 1.0


# -------------------------
# PREPARE OBJECT POINTS
# -------------------------

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob(image_folder)

print(f"Found {len(images)} calibration images")


# -------------------------
# DETECT CHECKERBOARD
# -------------------------

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:

        objpoints.append(objp)
        imgpoints.append(corners)

        # draw corners for visualization
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Checkerboard")
        plt.axis("off")
        plt.show()

print(f"Valid checkerboard detections: {len(objpoints)}")


# -------------------------
# CALIBRATE CAMERA
# -------------------------

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)


# -------------------------
# UNDISTORT EXAMPLE IMAGE
# -------------------------

test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w,h), 1, (w,h)
)

undistorted = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)

# Show comparison
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
plt.title("Undistorted Image")
plt.axis("off")

plt.tight_layout()
plt.show()
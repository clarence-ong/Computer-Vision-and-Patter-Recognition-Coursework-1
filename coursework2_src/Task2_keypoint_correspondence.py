import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# MANUAL CORRESPONDENCE
# -----------------------------
manual_coords_img1 = []
manual_coords_img2 = []

def onclick(event, coords):
    if event.xdata is not None and event.ydata is not None:
        coords.append((event.xdata, event.ydata))
        print("Point selected:", coords[-1])

def manual_selection(image_path, coords_list):
    try:
        img = plt.imread(image_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, coords_list))
        plt.title("Click corresponding points, close window when done")
        plt.show()
    except Exception as e:
        print("Manual selection skipped:", e)

# -----------------------------
# INPUT IMAGE PATHS
# -----------------------------
img1_path = r"C:\Users\User\OneDrive\Desktop\Imperial\Modules\Computer Vision\Courswork\Coursework 2\dataset\FD\with_object\FD_with_object_01.jpeg"
img2_path = r"C:\Users\User\OneDrive\Desktop\Imperial\Modules\Computer Vision\Courswork\Coursework 2\dataset\FD\with_object\FD_with_object_04.jpeg"


# Manual point selection (interactive)
manual_selection(img1_path, manual_coords_img1)
manual_selection(img2_path, manual_coords_img2)

print("Manual coordinates for image1:", manual_coords_img1)
print("Manual coordinates for image2:", manual_coords_img2)
print("Number of manual correspondences:", len(manual_coords_img1))

# -----------------------------
# AUTOMATIC SIFT MATCHES
# -----------------------------
img1_color = cv2.imread(img1_path)
img2_color = cv2.imread(img2_path)
img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

print("Number of automatic matches detected:", len(good_matches))

# -----------------------------
# DRAW AUTOMATIC MATCHES
# -----------------------------
img_matches = cv2.drawMatches(img1_color, kp1,
                              img2_color, kp2,
                              good_matches[:50], None, flags=2)

# -----------------------------
# OVERLAY MANUAL POINTS
# -----------------------------
# Manual points on first image
for x, y in manual_coords_img1:
    cv2.circle(img_matches, (int(x), int(y)), 10, (0,255,0), -1)  # green

# Manual points on second image
for x, y in manual_coords_img2:
    # second image x offset in drawMatches
    offset_x = img1_color.shape[1]
    cv2.circle(img_matches, (int(x) + offset_x, int(y)), 10, (0,0,255), -1)  # red

# -----------------------------
# DISPLAY FINAL RESULT
# -----------------------------
plt.figure(figsize=(20,10))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Automatic SIFT Matches (coloured lines) + Manual Points (green/red dots)")
plt.show()
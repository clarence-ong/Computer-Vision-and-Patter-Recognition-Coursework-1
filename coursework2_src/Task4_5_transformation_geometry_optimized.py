#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent / "coursework2_dataset"
OUT = SCRIPT_DIR / "outputs_task45_optimized"
OUT.mkdir(parents=True, exist_ok=True)

HG_IMG1 = ROOT / "HG" / "with_object" / "HG_with_object_01.jpeg"
HG_IMG2 = ROOT / "HG" / "with_object" / "HG_with_object_04.jpeg"
FD_IMG1 = ROOT / "FD" / "with_object" / "FD_with_object_01.jpeg"
FD_IMG2 = ROOT / "FD" / "with_object" / "FD_with_object_04.jpeg"

K = np.array([[1428.81, 0.00, 1088.95],
              [0.00, 1393.74, 720.67],
              [0.00, 0.00, 1.00]], dtype=np.float64)
dist = np.array([0.1489, -0.8268, -0.0049, 0.00024, 1.6426], dtype=np.float64)
USE_UNDISTORT = False

RATIO = 0.75
RANSAC_REPROJ_THRESH_H = 4.0
RANSAC_REPROJ_THRESH_F = 1.5
MAX_FEATURES_TO_DRAW = 60
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OBJECT_ROI_FD1 = (520, 420, 860, 1070)

def read_rgb(path: Path):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if USE_UNDISTORT:
        h, w = img_bgr.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        img_bgr = cv2.undistort(img_bgr, K, dist, None, newK)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_rgb, gray

def detect_and_match(gray1, gray2, ratio=RATIO):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        raise RuntimeError("SIFT did not find enough descriptors.")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return kp1, kp2, good, pts1, pts2

def draw_matches(img1, kp1, img2, kp2, matches, out_path, title, max_draw=MAX_FEATURES_TO_DRAW, mask=None):
    if mask is not None:
        chosen = [m for m, keep in zip(matches, mask.ravel().tolist()) if keep]
    else:
        chosen = matches
    chosen = chosen[:max_draw]
    canvas = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), kp1,
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), kp2,
        chosen, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def estimate_homography(pts1, pts2):
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_REPROJ_THRESH_H)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H, mask

def estimate_fundamental(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, RANSAC_REPROJ_THRESH_F, 0.99)
    if F is None:
        raise RuntimeError("Fundamental matrix estimation failed.")
    return F, mask

def project_points_homography(H, pts):
    pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T
    proj = H @ pts_h
    proj = proj[:2] / proj[2:]
    return proj.T

def plot_homography_projection(img1, img2, pts1, pts2, H, mask, out_path):
    in1 = pts1[mask.ravel() == 1]
    in2 = pts2[mask.ravel() == 1]
    proj = project_points_homography(H, in1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img1)
    ax[0].scatter(in1[:, 0], in1[:, 1], s=16)
    ax[0].set_title("HG image 1: inlier keypoints", fontsize=11)
    ax[0].axis("off")
    ax[1].imshow(img2)
    ax[1].scatter(in2[:, 0], in2[:, 1], s=18, label="matched inliers")
    ax[1].scatter(proj[:, 0], proj[:, 1], s=18, marker="x", label="projected from image 1")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("HG image 2: measured vs projected points", fontsize=11)
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def line_endpoints_from_abc(line, width, height):
    a, b, c = line
    points = []
    if abs(b) > 1e-8:
        y = -c / b
        if 0 <= y < height:
            points.append((0, int(y)))
        y = -(c + a * (width - 1)) / b
        if 0 <= y < height:
            points.append((width - 1, int(y)))
    if abs(a) > 1e-8:
        x = -c / a
        if 0 <= x < width:
            points.append((int(x), 0))
        x = -(c + b * (height - 1)) / a
        if 0 <= x < width:
            points.append((int(x), height - 1))
    uniq = []
    for p in points:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) >= 2:
        return uniq[0], uniq[1]
    return None, None

def draw_epilines(img1, img2, pts1, pts2, F, mask, out_path, n_draw=12):
    in1 = pts1[mask.ravel() == 1]
    in2 = pts2[mask.ravel() == 1]
    idx = np.arange(len(in1))
    np.random.shuffle(idx)
    idx = idx[:min(n_draw, len(idx))]
    s1 = in1[idx]
    s2 = in2[idx]
    lines2 = cv2.computeCorrespondEpilines(s1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    h2, w2 = img2.shape[:2]
    canvas2 = img2.copy()
    for p2, line in zip(s2, lines2):
        color = tuple(np.random.rand(3,))
        p_start, p_end = line_endpoints_from_abc(line, w2, h2)
        if p_start is not None:
            canvas2 = cv2.line(
                canvas2.copy(), p_start, p_end,
                color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)),
                thickness=2
            )
        canvas2 = cv2.circle(
            canvas2.copy(), tuple(np.int32(p2)), 6,
            color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)),
            thickness=-1
        )
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img1)
    ax[0].scatter(s1[:, 0], s1[:, 1], s=20)
    ax[0].set_title("FD image 1: selected inlier points", fontsize=11)
    ax[0].axis("off")
    ax[1].imshow(canvas2)
    ax[1].set_title("FD image 2: corresponding epipolar lines", fontsize=11)
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def null_space_epipole(F):
    _, _, vt = np.linalg.svd(F)
    e = vt[-1]
    e = e / e[-1]
    return e

def intersection_homogeneous(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-8:
        return None
    return p / p[2]

def detect_line_segments(gray):
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi / 180, threshold=80,
                            minLineLength=120, maxLineGap=15)
    segments = []
    if lines is None:
        return segments
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        segments.append(((x1, y1, x2, y2), angle, length))
    return segments

def segment_to_line(seg):
    x1, y1, x2, y2 = seg
    return np.cross([x1, y1, 1.0], [x2, y2, 1.0])

def robust_vanishing_point(segments, angle_range):
    chosen = []
    for seg, angle, length in segments:
        a = angle
        while a < -90:
            a += 180
        while a >= 90:
            a -= 180
        if angle_range[0] <= a <= angle_range[1]:
            chosen.append((seg, length))
    chosen = sorted(chosen, key=lambda x: -x[1])[:12]
    lines = [segment_to_line(seg) for seg, _ in chosen]
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p = intersection_homogeneous(lines[i], lines[j])
            if p is not None and np.all(np.isfinite(p)):
                points.append(p[:2])
    if len(points) == 0:
        return None, [seg for seg, _ in chosen]
    points = np.array(points)
    vp = np.median(points, axis=0)
    return np.array([vp[0], vp[1], 1.0]), [seg for seg, _ in chosen]

def _extended_canvas(img, margin_ratio=0.22):
    h, w = img.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    canvas = np.ones((h + 2 * my, w + 2 * mx, 3), dtype=np.uint8) * 255
    canvas[my:my + h, mx:mx + w] = img
    return canvas, mx, my

def _transform_line_to_canvas(line, dx, dy):
    a, b, c = line
    return np.array([a, b, c - a * dx - b * dy], dtype=np.float64)

def _draw_point_or_arrow(ax, point, label, color, w, h, dx, dy):
    x, y = float(point[0]), float(point[1])
    cx, cy = x + dx, y + dy
    inside = (0 <= x < w) and (0 <= y < h)
    if inside:
        ax.scatter(cx, cy, s=80, marker="x", c=[color], linewidths=2)
        ax.text(cx + 10, cy + 10, label, color=color, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))
    else:
        x0 = np.clip(x, 0, w - 1)
        y0 = np.clip(y, 0, h - 1)
        ax.annotate("", xy=(cx, cy), xytext=(x0 + dx, y0 + dy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.scatter(cx, cy, s=80, marker="x", c=[color], linewidths=2)
        ax.text(cx + 10, cy + 10, f"{label} (outside image)", color=color, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

def draw_geometry_overlay_better(img, F, out_path):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    segments = detect_line_segments(edges)
    vp1, segs1 = robust_vanishing_point(segments, (-25, 25))
    vp2, segs2 = robust_vanishing_point(segments, (55, 89))
    e2 = null_space_epipole(F.T)

    h, w = img.shape[:2]
    canvas, dx, dy = _extended_canvas(img, margin_ratio=0.22)
    ch, cw = canvas.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(canvas)

    for seg in segs1[:8]:
        x1, y1, x2, y2 = seg
        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], linewidth=1.8)
    for seg in segs2[:8]:
        x1, y1, x2, y2 = seg
        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], linewidth=1.8)

    if vp1 is not None and vp2 is not None:
        horizon = np.cross(vp1, vp2)
        horizon_canvas = _transform_line_to_canvas(horizon, dx, dy)
        p_start, p_end = line_endpoints_from_abc(horizon_canvas, cw, ch)
        if p_start is not None:
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
                    linewidth=2.5, label="horizon")

    if vp1 is not None:
        _draw_point_or_arrow(ax, vp1[:2], "vanishing point 1", "tab:blue", w, h, dx, dy)
    if vp2 is not None:
        _draw_point_or_arrow(ax, vp2[:2], "vanishing point 2", "tab:orange", w, h, dx, dy)
    _draw_point_or_arrow(ax, e2[:2], "epipole", "red", w, h, dx, dy)

    ax.plot([dx, dx + w, dx + w, dx, dx], [dy, dy, dy + h, dy + h, dy],
            color="white", linewidth=1.5)

    ax.set_title("FD image 2: epipole, vanishing points and horizon", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def symmetric_epipolar_error(F, pts1, pts2):
    pts1h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    pts2h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)
    l2 = (F @ pts1h.T).T
    l1 = (F.T @ pts2h.T).T
    num = np.sum(pts2h * l2, axis=1) ** 2
    den = l2[:, 0] ** 2 + l2[:, 1] ** 2 + l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-8
    return num / den

def homography_reproj_error(H, pts1, pts2):
    proj = project_points_homography(H, pts1)
    return np.linalg.norm(proj - pts2, axis=1)

def image_extent(pts1, pts2):
    allpts = np.vstack([pts1, pts2])
    xmin, ymin = np.min(allpts, axis=0)
    xmax, ymax = np.max(allpts, axis=0)
    return xmin, ymin, xmax, ymax

def inject_random_outliers(pts1, pts2, frac):
    n = len(pts1)
    k = int(round(frac * n))
    cur1 = pts1.copy()
    cur2 = pts2.copy()
    idx = np.random.choice(n, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    xmin, ymin, xmax, ymax = image_extent(pts1, pts2)
    rand1 = np.column_stack([
        np.random.uniform(xmin, xmax, size=k),
        np.random.uniform(ymin, ymax, size=k)
    ]).astype(np.float32)
    rand2 = np.column_stack([
        np.random.uniform(xmin, xmax, size=k),
        np.random.uniform(ymin, ymax, size=k)
    ]).astype(np.float32)
    if k > 0:
        cur1[idx] = rand1
        cur2[idx] = rand2
    return cur1, cur2

def outlier_tolerance_curve(model_name, clean_pts1, clean_pts2, base_estimator, base_error_fn, out_path,
                            n_trials=30, max_outlier_frac=0.8):
    fractions = np.linspace(0.0, max_outlier_frac, 9)
    M0, mask0 = base_estimator(clean_pts1, clean_pts2)
    baseline_err = base_error_fn(M0, clean_pts1, clean_pts2)
    baseline_med = float(np.median(baseline_err))
    baseline_inliers = int(mask0.sum())

    success_rates = []
    median_errs = []
    median_inlier_fracs = []

    for frac in fractions:
        successes = 0
        errs = []
        inlier_fracs = []
        for _ in range(n_trials):
            cur1, cur2 = inject_random_outliers(clean_pts1, clean_pts2, frac)
            try:
                M, mask = base_estimator(cur1, cur2)
                est_inliers = int(mask.sum())
                inlier_fracs.append(est_inliers / len(cur1))
                err_clean = base_error_fn(M, clean_pts1, clean_pts2)
                med_clean = float(np.median(err_clean))
                errs.append(med_clean)
                err_ratio = med_clean / max(baseline_med, 1e-6)
                inlier_ratio = est_inliers / max(baseline_inliers, 1)
                if model_name == "H":
                    ok = (err_ratio < 3.0) and (inlier_ratio > 0.35)
                else:
                    ok = (err_ratio < 4.0) and (inlier_ratio > 0.25)
                successes += int(ok)
            except Exception:
                errs.append(np.nan)
                inlier_fracs.append(0.0)
        success_rates.append(successes / n_trials)
        median_errs.append(np.nanmedian(errs))
        median_inlier_fracs.append(np.median(inlier_fracs))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(fractions * 100, success_rates, marker="o")
    ax[0].set_xlabel("Injected outlier fraction (%)")
    ax[0].set_ylabel("Successful estimation rate")
    ax[0].set_title(f"Outlier tolerance for {model_name}")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(fractions * 100, median_errs, marker="o", label="median clean-set error")
    ax[1].plot(fractions * 100, median_inlier_fracs, marker="s", label="median estimated inlier fraction")
    ax[1].set_xlabel("Injected outlier fraction (%)")
    ax[1].set_title(f"Error / consensus trend for {model_name}")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    tol_idx = 0
    for i, sr in enumerate(success_rates):
        if sr >= 0.8:
            tol_idx = i
    tolerated = float(fractions[tol_idx] * 100)
    return {
        "baseline_med_error": baseline_med,
        "baseline_inliers": baseline_inliers,
        "tolerated_outlier_percent_at_80_success": tolerated,
    }

def rectify_pair(img1_rgb, img2_rgb, pts1, pts2, F):
    h, w = img1_rgb.shape[:2]
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w, h))
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed.")
    rect1 = cv2.warpPerspective(img1_rgb, H1, (w, h))
    rect2 = cv2.warpPerspective(img2_rgb, H2, (w, h))
    rect_pts1 = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H1).reshape(-1, 2)
    rect_pts2 = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H2).reshape(-1, 2)
    return rect1, rect2, rect_pts1, rect_pts2, H1, H2

def plot_rectified_pair(rect1, rect2, out_path):
    h = min(rect1.shape[0], rect2.shape[0])
    canvas = np.ones((h, rect1.shape[1] + rect2.shape[1], 3), dtype=np.uint8) * 255
    canvas[:rect1.shape[0], :rect1.shape[1]] = rect1
    canvas[:rect2.shape[0], rect1.shape[1]:] = rect2
    plt.figure(figsize=(14, 6))
    plt.imshow(canvas)
    for y in np.linspace(40, h - 40, 10):
        plt.axhline(y=y, linewidth=1)
    plt.title("Stereo-rectified FD pair with horizontal epipolar lines", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def compute_disparity(rect1, rect2):
    g1 = cv2.cvtColor(rect1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(rect2, cv2.COLOR_RGB2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16 * 10, blockSize=5,
        P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2, disp12MaxDiff=1,
        uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return matcher.compute(g1, g2).astype(np.float32) / 16.0

def roi_after_homography(rect_shape, H, roi):
    x1, y1, x2, y2 = roi
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    w = rect_shape[1]
    h = rect_shape[0]
    warped[:, 0] = np.clip(warped[:, 0], 0, w - 1)
    warped[:, 1] = np.clip(warped[:, 1], 0, h - 1)
    return warped.astype(int)

def plot_disparity_and_depth(rect1, disparity, H1, out_path):
    disp_vis = disparity.copy()
    disp_vis[disp_vis <= 0] = np.nan
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0.5
    depth[valid] = 1.0 / disparity[valid]
    depth[~valid] = np.nan
    roi_poly = roi_after_homography(rect1.shape[:2], H1, OBJECT_ROI_FD1)
    poly = np.vstack([roi_poly, roi_poly[0]])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(rect1)
    ax[0].plot(poly[:, 0], poly[:, 1], linewidth=2)
    ax[0].set_title("Rectified left image + object ROI", fontsize=11)
    ax[0].axis("off")

    im1 = ax[1].imshow(disp_vis)
    ax[1].plot(poly[:, 0], poly[:, 1], linewidth=2)
    ax[1].set_title("Disparity map", fontsize=11)
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(depth)
    ax[2].plot(poly[:, 0], poly[:, 1], linewidth=2)
    ax[2].set_title("Relative depth map (1/disparity)", fontsize=11)
    ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def main():
    hg1, hg1_gray = read_rgb(HG_IMG1)
    hg2, hg2_gray = read_rgb(HG_IMG2)
    kp1, kp2, matches_hg, pts1_hg, pts2_hg = detect_and_match(hg1_gray, hg2_gray)
    H, maskH = estimate_homography(pts1_hg, pts2_hg)
    draw_matches(hg1, kp1, hg2, kp2, matches_hg,
                 OUT / "task4_hg_matches_inliers.png",
                 "Task 4.1: HG pair inlier matches used for homography",
                 mask=maskH)
    plot_homography_projection(hg1, hg2, pts1_hg, pts2_hg, H, maskH,
                               OUT / "task4_homography_projection.png")

    fd1, fd1_gray = read_rgb(FD_IMG1)
    fd2, fd2_gray = read_rgb(FD_IMG2)
    kp3, kp4, matches_fd, pts1_fd, pts2_fd = detect_and_match(fd1_gray, fd2_gray)
    F, maskF = estimate_fundamental(pts1_fd, pts2_fd)
    draw_matches(fd1, kp3, fd2, kp4, matches_fd,
                 OUT / "task4_fd_matches_inliers.png",
                 "Task 4.2: FD pair inlier matches used for fundamental matrix",
                 mask=maskF)
    draw_epilines(fd1, fd2, pts1_fd, pts2_fd, F, maskF,
                  OUT / "task4_epipolar_lines.png")
    draw_geometry_overlay_better(fd2, F,
                                 OUT / "task4_epipole_vanishing_horizon.png")

    in_hg1 = pts1_hg[maskH.ravel() == 1]
    in_hg2 = pts2_hg[maskH.ravel() == 1]
    summary_H = outlier_tolerance_curve(
        "H", in_hg1, in_hg2, estimate_homography, homography_reproj_error,
        OUT / "task4_outlier_tolerance_h.png"
    )

    in_fd1 = pts1_fd[maskF.ravel() == 1]
    in_fd2 = pts2_fd[maskF.ravel() == 1]
    summary_F = outlier_tolerance_curve(
        "F", in_fd1, in_fd2, estimate_fundamental, symmetric_epipolar_error,
        OUT / "task4_outlier_tolerance_f.png"
    )

    rect1, rect2, _, _, H1, _ = rectify_pair(fd1, fd2, in_fd1, in_fd2, F)
    plot_rectified_pair(rect1, rect2, OUT / "task5_rectified_pair.png")
    disparity = compute_disparity(rect1, rect2)
    plot_disparity_and_depth(rect1, disparity, H1, OUT / "task5_disparity_depth.png")

    print("Done. Saved outputs to:", OUT.resolve())
    print("Homography H:\n", H)
    print("Fundamental matrix F:\n", F)
    print("HG matches / inliers:", len(pts1_hg), "/", int(maskH.sum()))
    print("FD matches / inliers:", len(pts1_fd), "/", int(maskF.sum()))
    print("Homography baseline median error:", summary_H["baseline_med_error"])
    print("Fundamental baseline median error:", summary_F["baseline_med_error"])
    print("Approx. H outlier tolerance at >=80% success:",
          summary_H["tolerated_outlier_percent_at_80_success"], "%")
    print("Approx. F outlier tolerance at >=80% success:",
          summary_F["tolerated_outlier_percent_at_80_success"], "%")

if __name__ == "__main__":
    main()

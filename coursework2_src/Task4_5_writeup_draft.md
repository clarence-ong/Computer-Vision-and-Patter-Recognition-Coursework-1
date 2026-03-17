# Task 4–5 write-up draft

## Quick check of Tasks 1–3
**Task 1** is covered: the report states that both FD and HG sequences were collected, with and without the object, and the appendix shows the full dataset. This matches the coursework requirement.
**Task 2** is mostly covered: the report compares manual and automatic correspondences and gives both a qualitative conclusion and a match count. One improvement would be to state how the six manual points were chosen and to report a simple quantitative quality measure, such as the fraction of automatic matches visually judged correct.
**Task 3** is mostly covered: the intrinsic matrix and distortion coefficients are reported, and an undistorted image is shown. However, the script does not actually call `cv2.cornerSubPix()` even though the text says it does, so the code and report should be made consistent. It would also help to report the number of valid checkerboard detections and the mean reprojection error.

## Task 4: Transformation Estimation
For the HG sequence, a pair of images was selected and SIFT keypoints were matched using FLANN. A homography was then estimated with RANSAC. Figure X shows the inlier matches used for estimation. Figure Y compares the true matched keypoints in the second image with the positions obtained by projecting the first-image keypoints through the estimated homography. The projected points align closely with the measured correspondences, indicating that the planar mapping assumption for the HG sequence is reasonable.

For the FD sequence, a second pair of images was matched using the same SIFT + FLANN pipeline, followed by RANSAC estimation of the fundamental matrix. Figure Z shows selected inlier keypoints in the first image and their corresponding epipolar lines in the second image. The matched points lie close to their corresponding epipolar lines, which indicates that the estimated fundamental matrix is geometrically consistent.

To visualise the scene geometry further, the epipole was computed as the null-space of the estimated fundamental matrix, and two dominant vanishing points were estimated from line segments detected on the checkerboard. The horizon line was then obtained as the line passing through the two vanishing points. In the resulting visualisation, the horizon is consistent with the checkerboard orientation and the perspective of the scene.

To study robustness to mismatches, the inlier correspondence sets were progressively corrupted by replacing an increasing fraction of correspondences with random mismatches. For each corruption level, the estimation was repeated several times and counted as successful if the median geometric error remained below a fixed threshold. The results show that both homography and fundamental matrix estimation remain stable under a moderate proportion of outliers, but the success rate falls rapidly once the outlier fraction becomes large. This is consistent with the expected behaviour of RANSAC-based estimation.

## Task 5: 3D Geometry
Using the inlier correspondences and the estimated fundamental matrix from the FD pair, uncalibrated stereo rectification was performed. Figure A shows the rectified image pair with horizontal guide lines. After rectification, corresponding scene points are expected to lie on the same image rows, which simplifies dense matching.

A dense disparity map was then estimated using StereoSGBM. From the disparity image, a relative depth map was obtained using the inverse relationship between depth and disparity. The object is visible as a region of stronger disparity than the background plane, while low-texture and occluded areas remain noisy. Since the reconstruction is based on an uncalibrated two-view pair and no absolute baseline is known, the depth map should be interpreted as a relative depth estimate rather than a metrically accurate 3D reconstruction.

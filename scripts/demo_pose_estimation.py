""" Demo script for pose estimation. """

import cv2
from os.path import exists
from src.pose_estimation import estimate_pose_from_image, draw_pose
from src.utils import wrapped_imread


#image = wrapped_imread("pose_estimation/testdata/2019-Sal24.jpg")
image = wrapped_imread("src/pose_estimation/testdata/2022-Sal59.jpg")
results, success = estimate_pose_from_image(image)
print(results)


draw_pose(image, results)

cv2.imshow("image", image)

suffix = 0
filename = f"src/pose_estimation/testdata/result{suffix}.jpg"
while exists(filename):
    suffix += 1
    filename = f"src/pose_estimation/testdata/result{suffix}.jpg"
cv2.imwrite(filename, image)
cv2.waitKey(0)

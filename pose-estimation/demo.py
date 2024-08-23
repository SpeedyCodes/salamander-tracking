import cv2
from os.path import exists
from estimate import estimate_pose_from_image


#image = cv2.imread("pose-estimation/testdata/2019-Sal24.jpg")
image = cv2.imread("pose-estimation/testdata/2022-Sal59.jpg")
results = estimate_pose_from_image(image)
print(results)
for body_part_name, (x, y, confidence) in results.items():
    print(f"{body_part_name}: ({x}, {y}), confidence: {confidence}")
    if confidence < 0.7:
        continue
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(image, body_part_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("image", image)

suffix = 0
filename = f"pose-estimation/testdata/result{suffix}.jpg"
while exists(filename):
    suffix += 1
    filename = f"pose-estimation/testdata/result{suffix}.jpg"
cv2.imwrite(filename, image)
cv2.waitKey(0)

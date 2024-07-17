import cv2 as cv
from utils.heic_imread_wrapper import wrapped_imread


def draw_rectangles(img, rectangles):
    line_color = (0, 255, 0)
    line_type = cv.LINE_4
    for (x, y, width, height) in rectangles:
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv.rectangle(img, top_left, bottom_right, line_color, lineType=line_type)

    return img


cascade = cv.CascadeClassifier('../training/haar_cascade/cascade/cascade.xml')
#input = wrapped_imread('../input/2024/IMG_3023.jpeg')
input = wrapped_imread('../input/2024/IMG_3024.jpeg')
#input = wrapped_imread('../input/2021/2021-Sal09.jpg')
#input = wrapped_imread('../input/2019/2019-Sal18.jpg')

rectangles = cascade.detectMultiScale(input)
detection_image = draw_rectangles(input, rectangles)

cv.imwrite('detection.jpg', detection_image)

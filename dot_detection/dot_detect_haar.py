import cv2 as cv
from utils.heic_imread_wrapper import wrapped_imread
from isolate_salamander import isolate_salamander
import threading


def draw_rectangles(img, rectangles):
    line_color = (0, 0, 255)
    line_type = cv.LINE_4
    for (x, y, width, height) in rectangles:
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv.rectangle(img, top_left, bottom_right, line_color, lineType=line_type)

    return img


lock = threading.Lock()
cascade = cv.CascadeClassifier('training/haar_cascade/cascade/cascade.xml')


def dot_detect_haar(image):
    lock.acquire()
    image = isolate_salamander(image)
    output = cascade.detectMultiScale(image)
    lock.release()
    return output

if __name__ == '__main__':
    inputs = [wrapped_imread('input/2024/IMG_3023.jpeg'),
              wrapped_imread('input/2024/IMG_3024.jpeg'),
              wrapped_imread('input/2021/2021-Sal09.jpg'),
              wrapped_imread('input/2019/2019-Sal18.jpg')]

    for index, input in enumerate(inputs):
        isolated = isolate_salamander(input)
        rectangles = cascade.detectMultiScale(isolated)
        # give the annotation file line
        print(f' {len(rectangles)}{ "".join([f" {x} {y} {width} {height}" for (x, y, width, height) in rectangles])}')

        detection_image = draw_rectangles(input, rectangles)

        cv.imwrite(f'detection{index}.jpg', detection_image)

import cv2 as cv
from numpy import ndarray

from src.utils import wrapped_imread
from src.preprocessing import isolate_salamander
import threading


def draw_rectangles(img: ndarray, rectangles: ndarray) -> ndarray:
    """
    Draw rectangles on an image
    :param img: the image
    :param rectangles: the rectangles
    :return: the drawn rectangles
    """
    line_color = (0, 0, 255)
    line_type = cv.LINE_4
    for (x, y, width, height) in rectangles:
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv.rectangle(img, top_left, bottom_right, line_color, lineType=line_type)

    return img


lock = threading.Lock()
cascade = cv.CascadeClassifier('training/haar_cascade/cascade/cascade.xml')


def dot_detect_haar(image: ndarray) -> ndarray:
    """
    Perform Dot Detection using Haar Cascade
    :param image: input image
    :return: the detected points
    """
    lock.acquire()
    output = cascade.detectMultiScale(image)
    lock.release()
    return output


def draw_dots(image: ndarray, rectangles: ndarray) -> ndarray:
    """
    Draw dots on an image from rectangles
    :param image: the image
    :param dots: the rectangles
    :return: the drawn dots
    """
    new_image = image.copy()
    for rectangle in rectangles:
        start_point = (rectangle[0], rectangle[1])
        end_point = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
        cv.rectangle(new_image, start_point, end_point, (255, 0, 0), 2)
    return new_image


if __name__ == '__main__':
    inputs = [wrapped_imread('input/2024/IMG_3023.jpeg'),
              wrapped_imread('input/2024/IMG_3024.jpeg'),
              wrapped_imread('input/2021/2021-Sal09.jpg'),
              wrapped_imread('input/2019/2019-Sal18.jpg')]

    for index, input in enumerate(inputs):
        isolated, _ = isolate_salamander(input)
        rectangles = cascade.detectMultiScale(isolated)
        # give the annotation file line
        print(f' {len(rectangles)}{ "".join([f" {x} {y} {width} {height}" for (x, y, width, height) in rectangles])}')

        detection_image = draw_rectangles(input, rectangles)

        cv.imwrite(f'detection{index}.jpg', detection_image)

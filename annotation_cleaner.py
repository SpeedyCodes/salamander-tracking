from utils import read_annotation_file
import cv2
from Generating_extra_data import resize_with_aspect_ratio

file_to_clean = 'benchmark_annos/previously_good.txt'

images, rectangles = read_annotation_file(file_to_clean)

desired_height = 1000

new_lines = []

# loop through all images, showing them and allowing deletion of false positives by clicking inside of them

for i, (image, rects) in enumerate(zip(images, rectangles)):

    scaling_factor = desired_height / image.shape[0]

    while True:
        edited_image = image.copy()
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(edited_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        def draw_circle(event, X, Y, flags, param):
            if event == 1:
                X /= scaling_factor
                Y /= scaling_factor
                for i in range(len(rects), 0, -1):
                    x, y, w, h = rects[i-1]
                    if x < X < x + w and y < Y < y + h:
                        rects.pop(i-1)
                        break

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

        edited_image = resize_with_aspect_ratio(edited_image, height=desired_height)

        cv2.imshow("image", edited_image)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('r'):
            print('reloaded')
        if key & 0xFF == ord('n'):
            print('n')
            newline = f'{len(rects)}'
            for j, rect in enumerate(rects):
                x, y, w, h = rect
                newline += f' {x} {y} {w} {h}'
            new_lines.append(newline)
            break

# read the filenames
with open(file_to_clean, 'r') as file:
    for index, line in enumerate(file.readlines()):
        filename = line.split(' ')[0]
        new_lines[index] = filename + " " + new_lines[index] + "\n"
    file.close()

# write the changed lines
with open(file_to_clean, 'w') as file:
    for line in new_lines:
        file.write(line)
    file.close()


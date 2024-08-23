from math import floor, ceil
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import cv2

class CoordinateTransformer:
    """
    This class is used to transform a point in the image corresponding to the transformation
    needed to transform the spine to a straight line.
    """
    def __init__(self, datapoints: Dict[str, Tuple[int, int, float]]):
        """
        :param datapoints: A dictionary containing the raw datapoints of the pose estimation (name: (x, y, confidence))
        """

        self.datapoints = datapoints
        self.x_given = np.array([])
        self.y_given = np.array([])
        # not all datapoints are part of the spine
        self.used_points = ["spine_highest", "spine_high", "spine_middle", "spine_low", "spine_lowest"]

        self.x_remapped = np.array([0] * len(self.used_points))
        self.y_remapped = np.array([0] * len(self.used_points))

        for body_part_name, (x, y, confidence) in datapoints.items():
            if body_part_name in self.used_points:
                self.x_given = np.append(self.x_given, x)
                self.y_given = np.append(self.y_given, y)

        self.xPoly = self._get_interpolation_for_axis(self.x_given)
        self.yPoly = self._get_interpolation_for_axis(self.y_given)

        self.middle_point_t = self.used_points.index("spine_middle")

        self._transform_spine()

    def _get_interpolation_for_axis(self, datapoints):
        t = range(len(datapoints))
        x = datapoints
        #return interp1d(t, x, kind='quadratic')
        return CubicSpline(t, x, bc_type='natural')

    def show_interpolation(self, image):

        t_min = 0
        t_max = len(self.x_given) - 1

        lspace = np.linspace(t_min, t_max, 100)
        draw_x = self.xPoly(lspace)
        draw_y = self.yPoly(lspace)

        draw_points = np.asarray([draw_x, draw_y]).T.astype(np.int32)
        cv2.polylines(image, [draw_points], False, (0, 0, 0))

        for x, y in zip(self.x_given, self.y_given):
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # draw the new points and line

        for x, y in zip(self.x_remapped, self.y_remapped):
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

        cv2.line(image, (int(self.x_remapped[0]), int(self.y_remapped[0])),
                 (int(self.x_remapped[4]), int(self.y_remapped[4])), (255, 0, 0))

        cv2.imshow("image", image)
        cv2.waitKey(0)


    def _transform_spine(self):
        # remaps the original data points of the spine to a straight line


        # middle point is the same
        self.x_remapped[self.middle_point_t] = self.x_given[self.middle_point_t]
        self.y_remapped[self.middle_point_t] = self.y_given[self.middle_point_t]

        for delta in [-1, 1]:
            # first do the inner points
            inner_point_t = self.middle_point_t + delta
            self.x_remapped[inner_point_t] = self.x_given[self.middle_point_t]
            # TODO this should probably be the arc length over the spline
            distance_between_points = np.sqrt((self.x_given[self.middle_point_t] - self.x_given[inner_point_t]) ** 2 + (
                    self.y_given[self.middle_point_t] - self.y_given[inner_point_t]) ** 2)
            self.y_remapped[inner_point_t] = self.y_given[self.middle_point_t] + delta * distance_between_points

            # continue with the outer points
            outer_point_t = self.middle_point_t + 2 * delta
            self.x_remapped[outer_point_t] = self.x_given[self.middle_point_t]
            # TODO this should probably be the arc length over the spline
            distance_between_points = np.sqrt((self.x_given[inner_point_t] - self.x_given[outer_point_t]) ** 2 + (
                    self.y_given[inner_point_t] - self.y_given[outer_point_t]) ** 2)
            self.y_remapped[outer_point_t] = self.y_given[inner_point_t] + delta * distance_between_points

    def transform(self, x, y):
        """
        Takes in a point in the image and transforms it along the straightened spine
        :param x:
        :param y:
        :return:
        """

        #  find the closest point in the interpolation
        t = np.linspace(0, len(self.x_given) - 1, 1000)
        x_t = self.xPoly(t)
        y_t = self.yPoly(t)
        distances = np.sqrt((x_t - x) ** 2 + (y_t - y) ** 2)
        closest_point_index = np.argmin(distances)
        closest_point = (int(x_t[closest_point_index]), int(y_t[closest_point_index]))
        distance = distances[closest_point_index]

        # compute the sign of the distance
        closest_point_t = t[closest_point_index]
        lower_t_bound = floor(closest_point_t)
        upper_t_bound = ceil(closest_point_t)
        lower_bound_coords = (self.xPoly(lower_t_bound), self.yPoly(lower_t_bound))
        upper_bound_coords = (self.xPoly(upper_t_bound), self.yPoly(upper_t_bound))
        lower_to_upper = upper_bound_coords[0] - lower_bound_coords[0], upper_bound_coords[1] - lower_bound_coords[1]
        lower_to_closest = x - lower_bound_coords[0], y - lower_bound_coords[1]

        sign = np.sign(np.cross(lower_to_closest, lower_to_upper))
        distance *= sign

        # get the length of spline between the closest point and the spine_middle
        spine_part_needed = closest_point_t / (len(self.used_points) - 1)
        full_spine_length = self.y_remapped.max() - self.y_remapped.min()
        spine_length = spine_part_needed * full_spine_length
        new_point_x = self.x_remapped[self.middle_point_t] + distance
        new_point_y = self.y_remapped.min() + spine_length

        # TODO removeme
        cv2.circle(image, closest_point, 5, (0, 255, 0), -1)
        cv2.line(image, (x, y), closest_point, (0, 255, 0))
        cv2.circle(image, (int(new_point_x), int(new_point_y)), 5, (0, 255, 255), -1)

        return new_point_x, new_point_y



if __name__ == "__main__":
    image = cv2.imread("pose-estimation/testdata/2022-Sal59.jpg")
    input = {'head_tip': (175, 34, 0.9991191029548645), 'left_shoulder': (166, 251, 0.9292972087860107), 'left_hand_middle': (77, 347, 0.5459177494049072), 'right_shoulder': (292, 255, 0.9942055344581604), 'right_hand_middle': (455, 347, 0.3259645402431488), 'left_pelvis': (263, 671, 0.9855242371559143), 'left_foot_middle': (154, 845, 0.44996482133865356), 'right_pelvis': (328, 571, 0.9363328814506531), 'right_foot_middle': (386, 463, 0.9253280162811279), 'tail_connection': (350, 714, 0.9914472103118896), 'tail_end': (179, 748, 0.025986261665821075), 'spine_highest': (227, 262, 0.9916161298751831), 'spine_high': (230, 317, 0.7306284308433533), 'spine_middle': (235, 410, 0.7062945365905762), 'spine_low': (249, 513, 0.9910240173339844), 'spine_lowest': (275, 595, 0.897484540939331)}

    point = 240, 550

    cv2.circle(image, point, 5, (0, 0, 255), -1)
    transformer = CoordinateTransformer(input)

    transformer.transform(*point)
    transformer.show_interpolation(image)

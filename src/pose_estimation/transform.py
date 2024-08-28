from math import floor, ceil
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy import integrate
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
        # not all datapoints are part of the spine
        self.used_points = ["spine_highest", "spine_high", "spine_middle", "spine_low", "spine_lowest"]

        self.x_remapped = np.array([0] * len(self.used_points))
        self.y_remapped = np.array([0] * len(self.used_points))

        x_given = []
        y_given = []
        for body_part_name in self.used_points:
            x, y, confidence = datapoints[body_part_name]
            x_given.append(x)
            y_given.append(y)

        self.x_given = np.array(x_given)
        self.y_given = np.array(y_given)

        self.x_poly = self._get_interpolation_for_axis(self.x_given)
        self.y_poly = self._get_interpolation_for_axis(self.y_given)
        self.x_poly_derivative = self.x_poly.derivative()
        self.y_poly_derivative = self.y_poly.derivative()
        self.arc_length_integrand = lambda t: np.sqrt(self.x_poly_derivative(t) ** 2 + self.y_poly_derivative(t) ** 2)

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
        draw_x = self.x_poly(lspace)
        draw_y = self.y_poly(lspace)

        draw_points = np.asarray([draw_x, draw_y]).T.astype(np.int32)
        cv2.polylines(image, [draw_points], False, (0, 0, 0))

        for x, y in zip(self.x_given, self.y_given):
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # draw the new points and line

        for x, y in zip(self.x_remapped, self.y_remapped):
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

        cv2.line(image, (int(self.x_remapped[0]), int(self.y_remapped[0])),
                 (int(self.x_remapped[-1]), int(self.y_remapped[-1])), (255, 0, 0))

        cv2.imshow("image", image)
        cv2.waitKey(0)


    def _transform_spine(self):
        # remaps the original data points of the spine to a straight line
        # not used anymore, just keep it for visualisation

        # middle point is the same
        self.x_remapped[0] = self.x_given[0]
        self.y_remapped[0] = self.y_given[0]

        # work downwards from the highest point
        delta = 1
        base_point_t = 0
        new_point_t = delta
        while 0 <= new_point_t < len(self.x_given):
            self.x_remapped[new_point_t] = self.x_remapped[base_point_t]
            distance_between_points = abs(self._arc_length(base_point_t, new_point_t))
            self.y_remapped[new_point_t] = self.y_remapped[base_point_t] + delta * distance_between_points

            base_point_t = new_point_t
            new_point_t += delta

    def _arc_length(self, t_start: int, t_end: int):
        return integrate.quad(self.arc_length_integrand, t_start, t_end)[0]

    def transform(self, x, y):
        """
        Takes in a point in the image and transforms it along the straightened spine
        :param x:
        :param y:
        :return:
        """

        #  find the closest point in the interpolation
        t = np.linspace(0, len(self.x_given) - 1, 1000)
        x_t = self.x_poly(t)
        y_t = self.y_poly(t)
        distances = np.sqrt((x_t - x) ** 2 + (y_t - y) ** 2)
        closest_point_index = np.argmin(distances)
        closest_point = (int(x_t[closest_point_index]), int(y_t[closest_point_index]))
        distance = distances[closest_point_index]

        # compute the sign of the distance
        closest_point_t = t[closest_point_index]
        lower_t_bound = floor(closest_point_t)
        upper_t_bound = ceil(closest_point_t)
        lower_bound_coords = (self.x_poly(lower_t_bound), self.y_poly(lower_t_bound))
        upper_bound_coords = (self.x_poly(upper_t_bound), self.y_poly(upper_t_bound))
        lower_to_upper = upper_bound_coords[0] - lower_bound_coords[0], upper_bound_coords[1] - lower_bound_coords[1]
        lower_to_closest = x - lower_bound_coords[0], y - lower_bound_coords[1]

        sign = np.sign(np.cross(lower_to_closest, lower_to_upper))
        distance *= sign

        # get the length of spline between the upper point and the spine_middle
        spine_length_needed = abs(self._arc_length(0, closest_point_t))
        # normalize the distances to the length of the spine
        full_spine_length = self._arc_length(0, len(self.x_given) - 1)
        new_point_x = distance / full_spine_length
        new_point_y = spine_length_needed / full_spine_length

        # TODO removeme
        cv2.circle(image, closest_point, 5, (0, 255, 0), -1)
        cv2.putText(image, "closest point on spine", closest_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(image, (x, y), closest_point, (0, 255, 0))
        cv2.circle(image, (int(new_point_x + self.x_remapped[0]), int(new_point_y + self.y_remapped[0])), 5, (0, 0, 0), -1)
        cv2.putText(image, "new", (int(new_point_x), int(new_point_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return new_point_x, new_point_y



if __name__ == "__main__":
    image = cv2.imread("src/pose_estimation/testdata/2022-Sal59.jpg")
    input = {'head_tip': (175, 34, 0.9991191029548645), 'left_shoulder': (166, 251, 0.9292972087860107), 'left_hand_middle': (77, 347, 0.5459177494049072), 'right_shoulder': (292, 255, 0.9942055344581604), 'right_hand_middle': (455, 347, 0.3259645402431488), 'left_pelvis': (263, 671, 0.9855242371559143), 'left_foot_middle': (154, 845, 0.44996482133865356), 'right_pelvis': (328, 571, 0.9363328814506531), 'right_foot_middle': (386, 463, 0.9253280162811279), 'tail_connection': (350, 714, 0.9914472103118896), 'tail_end': (179, 748, 0.025986261665821075), 'spine_highest': (227, 262, 0.9916161298751831), 'spine_high': (230, 317, 0.7306284308433533), 'spine_middle': (235, 410, 0.7062945365905762), 'spine_low': (249, 513, 0.9910240173339844), 'spine_lowest': (275, 595, 0.897484540939331)}

    point = 240, 550

    cv2.circle(image, point, 5, (0, 0, 255), -1)
    cv2.putText(image, "original", (point[0]-30, point[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    transformer = CoordinateTransformer(input)

    x, y =transformer.transform(*point)
    transformer.show_interpolation(image)

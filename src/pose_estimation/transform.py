from math import floor, ceil
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate
import cv2
from src.utils import ImageQuality
from config import pose_estimation_confidence

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

        missing_points = []

        x_given = []
        y_given = []
        # check which points we can readily use
        for body_part_name in self.used_points:
            if body_part_name not in datapoints:
                missing_points.append(body_part_name)
                continue
            x, y, confidence = datapoints[body_part_name]
            if confidence < pose_estimation_confidence:
                missing_points.append(body_part_name)
                continue
            x_given.append(x)
            y_given.append(y)

        self.image_quality = ImageQuality.GOOD

        if len(missing_points) > 0:
            # if one or two points are missing, we can interpolate them (except for the border points)
            # otherwise, the image is too bad
            if (len(missing_points) > 2 or self.used_points[0] in missing_points
                    or self.used_points[-1] in missing_points):
                self.image_quality = ImageQuality.MEDIUM
                return  # the image is too bad to interpolate, stop

        self.x_given = np.array(x_given)
        self.y_given = np.array(y_given)

        # create a first interpolated polynomial between the points
        self.x_poly = self._get_interpolation_for_axis(self.x_given)
        self.y_poly = self._get_interpolation_for_axis(self.y_given)

        interpolated_points = {}

        if len(missing_points) > 0:
            # interpolate missing body parts
            if len(missing_points) == 1:
                body_part_name = missing_points[0]
                inter_t = (self.used_points.index(body_part_name)-1+self.used_points.index(body_part_name))/2
                interpolated_points[self.used_points.index(body_part_name)] = inter_t
            else: # two missing parts (we don't allow more), either next to each other or one valid point inbetween
                missing_point_index_1 = self.used_points.index(missing_points[0])
                missing_point_index_2 = self.used_points.index(missing_points[1])
                if abs(missing_point_index_1 - missing_point_index_2) == 1: # right next to each other
                    previous_good_point = missing_point_index_1 - 1
                    interpolated_points[missing_point_index_1] = previous_good_point + 1/3
                    interpolated_points[missing_point_index_2] = previous_good_point + 2/3
                else: # one valid point inbetween
                    interpolated_points[missing_point_index_1] = missing_point_index_1 - 1/2
                    interpolated_points[missing_point_index_2] = len(self.x_given) -1 - 1/2



            for index, t in interpolated_points.items():
                this_x = self.x_poly(t)
                this_y = self.y_poly(t)
                self.x_given = np.insert(self.x_given, index, this_x)
                self.y_given = np.insert(self.y_given, index, this_y)

            # it's easier for transformation later if the polynomials are created with a fixed (max) number of datapoints, so
            # recompute the interpolation with inserted points
            self.x_poly = self._get_interpolation_for_axis(self.x_given)
            self.y_poly = self._get_interpolation_for_axis(self.y_given)

        # prepare to calculate arc length
        self.x_poly_derivative = self.x_poly.derivative()
        self.y_poly_derivative = self.y_poly.derivative()
        self.arc_length_integrand = lambda t: np.sqrt(self.x_poly_derivative(t) ** 2 + self.y_poly_derivative(t) ** 2)

        # for visualisation purposes
        self.x_remapped = np.array([0] * len(self.x_given))
        self.y_remapped = np.array([0] * len(self.x_given))
        self._transform_spine()

    def _get_interpolation_for_axis(self, datapoints: np.ndarray) -> CubicSpline:
        """
        Provides an interpolating piecewise polynomial for the given datapoints.
        :param datapoints: the datapoints to interpolate
        :return: the polynomial
        """
        t = range(len(datapoints))
        return CubicSpline(t, datapoints, bc_type='natural')

    def show_transformation(self, base_image: np.ndarray) -> np.ndarray:
        """
        Shows the transformation of the spine on the given image.
        :param base_image: image to draw on
        :return: the drawing of the transformation on the given image
        """

        image = base_image.copy()
        t_min = 0
        t_max = len(self.x_given) - 1

        lspace = np.linspace(t_min, t_max, 100)
        draw_x = self.x_poly(lspace)
        draw_y = self.y_poly(lspace)

        # draw the old points and line

        draw_points = np.asarray([draw_x, draw_y]).T.astype(np.int32)
        cv2.polylines(image, [draw_points], False, (0, 0, 0))

        for x, y in zip(self.x_given, self.y_given):
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # draw the new points and line

        for x, y in zip(self.x_remapped, self.y_remapped):
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

        cv2.line(image, (int(self.x_remapped[0]), int(self.y_remapped[0])),
                 (int(self.x_remapped[-1]), int(self.y_remapped[-1])), (255, 0, 0))

        return image


    def _transform_spine(self):
        # remaps the original data points of the spine to a straight line
        # not used anymore, just keep it for visualisation

        # highest point is the same
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
        """
        Calculates the arc length between two points on the interpolated polynomial.
        :param t_start: the start point
        :param t_end: the end point
        :return: the arc length
        """
        return integrate.quad(self.arc_length_integrand, t_start, t_end)[0]

    def transform(self, x, y):
        """
        Takes in a point in the image and transforms it along the straightened spine
        :param x:
        :param y:
        :return:
        """

        assert self.image_quality > ImageQuality.BAD

        #  find the closest point in the interpolation
        t = np.linspace(0, len(self.x_given) - 1, 1000)
        x_t = self.x_poly(t)
        y_t = self.y_poly(t)
        distances = np.sqrt((x_t - x) ** 2 + (y_t - y) ** 2)
        closest_point_index = np.argmin(distances)
        closest_point = (int(x_t[closest_point_index]), int(y_t[closest_point_index]))
        distance = distances[closest_point_index]

        # compute the sign of the distance between the real and interpolated point
        closest_point_t = t[closest_point_index]
        lower_t_bound = floor(closest_point_t)
        upper_t_bound = ceil(closest_point_t)
        lower_bound_coords = (self.x_poly(lower_t_bound), self.y_poly(lower_t_bound))
        upper_bound_coords = (self.x_poly(upper_t_bound), self.y_poly(upper_t_bound))
        lower_to_upper = upper_bound_coords[0] - lower_bound_coords[0], upper_bound_coords[1] - lower_bound_coords[1]
        lower_to_closest = x - lower_bound_coords[0], y - lower_bound_coords[1]

        sign = np.sign(np.cross(lower_to_closest, lower_to_upper))
        distance *= sign

        # put the new point along the straightened spine, offset along the x axis by the normalized (still signed) distance
        # get the length of spline between the upper point and the spine_middle
        spine_length_needed = abs(self._arc_length(0, closest_point_t))
        # normalize the distances to the length of the spine
        full_spine_length = self._arc_length(0, len(self.x_given) - 1)
        new_point_x = distance / full_spine_length
        new_point_y = spine_length_needed / full_spine_length

        return new_point_x, new_point_y



if __name__ == "__main__":
    image = cv2.imread("input/2022/2022-Sal59.jpg")
    input = {'head_tip': (175, 34, 0.9991191029548645), 'left_shoulder': (166, 251, 0.9292972087860107),
             'left_hand_middle': (77, 347, 0.5459177494049072), 'right_shoulder': (292, 255, 0.9942055344581604),
             'right_hand_middle': (455, 347, 0.3259645402431488), 'left_pelvis': (263, 671, 0.9855242371559143),
             'left_foot_middle': (154, 845, 0.44996482133865356), 'right_pelvis': (328, 571, 0.9363328814506531),
             'right_foot_middle': (386, 463, 0.9253280162811279), 'tail_connection': (350, 714, 0.9914472103118896),
             'tail_end': (179, 748, 0.025986261665821075), 'spine_highest': (227, 262, 0.9916161298751831),
             'spine_high': (230, 317, 0.7306284308433533), 'spine_middle': (235, 410, 0.7062945365905762),
             'spine_low': (249, 513, 0.9910240173339844), 'spine_lowest': (275, 595, 0.897484540939331)}

    point = 240, 550

    cv2.circle(image, point, 5, (0, 0, 255), -1)
    cv2.putText(image, "original", (point[0]-30, point[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    transformer = CoordinateTransformer(input)

    x, y =transformer.transform(*point)
    annotated_image = transformer.show_transformation(image)

    cv2.imshow("image", annotated_image)
    cv2.waitKey(0)

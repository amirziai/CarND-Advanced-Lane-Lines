from perspective import Perspective
from processing import *

import numpy as np
import cv2

# Config
ARROW_TIP_LENGTH = config.lane_detection['arrow_tip_length']
VERTICAL_OFFSET = config.lane_detection['vertical_offset']
HISTOGRAM_WINDOW = config.lane_detection['histogram_window']
POLYNOMIAL_COEFFICIENT = config.lane_detection['polynomial_coefficient']
LINE_SEGMENTS = config.lane_detection['line_segments']

# Took plenty of inspiration from:
# https://github.com/mimoralea/CarND-Advanced-Lane-Lines
# https://github.com/paul-o-alto/CarND-Advanced-Lane-Lines
# https://github.com/pkern90/CarND-advancedLaneLines


class Line:
    def __init__(self, n_images=1, x=None, y=None):
        self.n_images = n_images  # history to keep
        self.x_recent = []  # most recent x
        self.pixels = []  # # pixels added per image
        self.x_average = None  # average of x over the last n
        self.best_fit = None  # average polynomial coefficients
        self.current_coef = None  # current polynomial coefs
        self.current_coef_poly = None  # polynomial for the current fit
        self.best_fit_poly = None  # average of the last n polynomial
        self.radius = None  # radius
        self.line_base_pos = None  # distance from center
        self.diffs = np.array([0, 0, 0], dtype='float')  # delta in fit coefs between last and new fits
        self.found = False  # found in previous step
        self.xs = None  # x values for found line pixels
        self.ys = None
        if x:
            self.update(x, y)

    def update(self, x, y):
        self.xs = x
        self.ys = y

        self.pixels.append(len(self.xs))
        self.x_recent.extend(self.xs)

        if len(self.pixels) > self.n_images:
            n_x_to_remove = self.pixels.pop(0)
            self.x_recent = self.x_recent[n_x_to_remove:]

        self.x_average = np.mean(self.x_recent)

        self.current_coef = np.polyfit(self.xs, self.ys, 2)

        if self.best_fit is None:
            self.best_fit = self.current_coef
        else:
            self.best_fit = (self.best_fit * (self.n_images - 1) + self.current_coef) / self.n_images

        self.current_coef_poly = np.poly1d(self.current_coef)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def is_current_coef_parallel(self, other_line, threshold=(0, 0)):
        first_coefficient_delta = np.abs(self.current_coef[0] - other_line.current_coef[0])
        second_coefficient_delta = np.abs(self.current_coef[1] - other_line.current_coef[1])
        is_parallel = first_coefficient_delta < threshold[0] and second_coefficient_delta < threshold[1]

        return is_parallel

    def get_current_coef_distance(self, other_line):
        return np.abs(self.current_coef_poly(POLYNOMIAL_COEFFICIENT)
                      - other_line.current_coef_poly(POLYNOMIAL_COEFFICIENT))

    def get_best_fit_distance(self, other_line):
        return np.abs(self.best_fit_poly(POLYNOMIAL_COEFFICIENT) - other_line.best_fit_poly(POLYNOMIAL_COEFFICIENT))


class LaneDetector:
    def __init__(self, src, dst, n_images=1, calibration=None, line_segments=LINE_SEGMENTS, offset=0):
        self.n_images = n_images
        self.cam_calibration = calibration
        self.line_segments = line_segments
        self.image_offset = offset
        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.offset = 0.0
        self.perspective_src = src
        self.perspective_dst = dst
        self.perspective = Perspective(src, dst)
        self.dists = []

    @staticmethod
    def _acceptable_lanes(left, right):
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])
            return acceptable_lanes(new_left, new_right)

    def _check_lines(self, left_x, left_y, right_x, right_y):
        left_found, right_found = False, False

        if self._acceptable_lanes((left_x, left_y), (right_x, right_y)):
            left_found, right_found = True, True
        elif self.left_line and self.right_line:
            if self._acceptable_lanes((left_x, left_y), (self.left_line.ys, self.left_line.xs)):
                left_found = True
            if self._acceptable_lanes((right_x, right_y), (self.right_line.ys, self.right_line.xs)):
                right_found = True

        return left_found, right_found

    def _draw_info(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_curvature = 'Curvature: {}'.format(self.curvature)
        cv2.putText(image, text_curvature, (50, 50), font, 1, (255, 255, 255), 2)
        text_position = '{}m {} of center'.format(abs(self.offset), 'left' if self.offset < 0 else 'right')
        cv2.putText(image, text_position, (50, 100), font, 1, (255, 255, 255), 2)

    def _draw_overlay(self, image):
        overlay = np.zeros([*image.shape])
        mask = np.zeros([image.shape[0], image.shape[1]])
        lane_area = calculate_lane_area((self.left_line, self.right_line), image.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = self.perspective.inverse_transform(mask)
        overlay[mask == 1] = (255, 128, 0)
        selection = (overlay != 0)
        image[selection] = image[selection] * 0.3 + overlay[selection] * 0.7
        mask[:] = 0
        mask = draw_polynomial(mask, self.center_poly, 20, 255, 5, True, ARROW_TIP_LENGTH)
        mask = self.perspective.inverse_transform(mask)
        image[mask == 255] = (255, 75, 2)
        mask[:] = 0
        mask = draw_polynomial(mask, self.left_line.best_fit_poly, 5, 255)
        mask = draw_polynomial(mask, self.right_line.best_fit_poly, 5, 255)
        mask = self.perspective.inverse_transform(mask)
        image[mask == 255] = (255, 200, 2)

    def _process_history(self, image, left_found, right_found, left_x, left_y, right_x, right_y):
        if self.left_line and self.right_line:
            left_x, left_y = lane_detection_history(image, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = lane_detection_history(image, self.right_line.best_fit_poly, self.line_segments)

            left_found, right_found = self._check_lines(left_x, left_y, right_x, right_y)
        return left_found, right_found, left_x, left_y, right_x, right_y

    def _process_histogram(self, image, left_found, right_found, left_x, left_y, right_x, right_y):
        if not left_found:
            left_x, left_y = lane_detection_histogram(image, self.line_segments,
                                                      (self.image_offset, image.shape[1] // 2),
                                                      h_window=HISTOGRAM_WINDOW)
            left_x, left_y = remove_outliers(left_x, left_y)
        if not right_found:
            right_x, right_y = lane_detection_histogram(image, self.line_segments,
                                                        (image.shape[1] // 2, image.shape[1] - self.image_offset),
                                                        h_window=HISTOGRAM_WINDOW)
            right_x, right_y = remove_outliers(right_x, right_y)

        if not left_found or not right_found:
            left_found, right_found = self._check_lines(left_x, left_y, right_x, right_y)

        return left_found, right_found, left_x, left_y, right_x, right_y

    def _draw(self, image, original_image):
        if self.left_line and self.right_line:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = curvature(self.center_poly)
            self.offset = (image.shape[1] / 2 - self.center_poly(POLYNOMIAL_COEFFICIENT)) * 3.7 / 700
            self._draw_overlay(original_image)
            self._draw_info(original_image)

    def _update_lane_left(self, found, x, y):
        if found:
            if self.left_line:
                self.left_line.update(y=x, x=y)
            else:
                self.left_line = Line(self.n_images, y, x)

    def _update_lane_right(self, found, x, y):
        if found:
            if self.right_line:
                self.right_line.update(y=x, x=y)
            else:
                self.right_line = Line(self.n_images, y, x)

    def process_image(self, image):
        original_image = np.copy(image)

        image = self.cam_calibration.undistort(image)
        image = lane_mask(image, VERTICAL_OFFSET)
        image = self.perspective.transform(image)

        left_found = right_found = False
        left_x = left_y = right_x = right_y = []

        left_found, right_found, left_x, left_y, right_x, right_y = \
            self._process_history(image, left_found, right_found, left_x, left_y, right_x, right_y)
        left_found, right_found, left_x, left_y, right_x, right_y = \
            self._process_histogram(image, left_found, right_found, left_x, left_y, right_x, right_y)

        self._update_lane_left(left_found, left_x, left_y)
        self._update_lane_right(right_found, right_x, right_y)
        self._draw(image, original_image)

        return original_image

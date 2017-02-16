import config
import utils

import os
from scipy.misc import imread, imresize
import numpy as np
import glob
import cv2

IMAGE_SIZE = config.calibration['image_size']
CALIBRATION_IMAGE_SIZE = config.calibration['calibration_image_size']
CALIBRATION_PICKLE_FILE = config.calibration['calibration_pickle_file']
IMAGES_PATH = config.calibration['images_path']
CHESSBOARD_ROWS = config.calibration['chessboard_rows']
CHESSBOARD_COLS = config.calibration['chessboard_cols']


class Calibration:
    def __init__(self, image_size=IMAGE_SIZE, calibration_file=CALIBRATION_PICKLE_FILE):
        # Get camera calibration
        points_object, points_image = (utils.unpickle(calibration_file) if os.path.exists(calibration_file)
                                       else self._calibrate())
        # Get mtx and dist for undistorting new images
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(points_object, points_image, image_size, None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    @staticmethod
    def _calibrate(images_path=IMAGES_PATH, chessboard_rows=CHESSBOARD_ROWS, chessboard_cols=CHESSBOARD_COLS,
                   image_size=CALIBRATION_IMAGE_SIZE, calibration_pickle_file=CALIBRATION_PICKLE_FILE):
        obj = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
        obj[:, :2] = np.mgrid[:chessboard_cols, :chessboard_rows].T.reshape(-1, 2)

        points_object = []
        points_image = []

        images = glob.glob(images_path)

        for image in images:
            image_array = imread(image)
            if image_array.shape != image_size:
                image_array = imresize(image_array, image_size)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

            if ret:
                points_object.append(obj)
                points_image.append(corners)

        calibration = (points_object, points_image)
        utils.pickle(calibration, calibration_pickle_file)
        return calibration

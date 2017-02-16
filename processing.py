import config

from itertools import islice
import cv2
import numpy as np
from scipy import signal

# Config
KERNEL_SIZE = config.processing['kernel_size']
FILTER_2D = config.processing['filter_2d']
FILTER_2D_THRESHOLD = config.processing['filter_2d_threshold']
HISTOGRAM_PEAKS = config.processing['histogram_peaks']
OUTLIER_PERCENTILE = config.processing['outlier_percentile']
PARALLEL_THRESHOLD = config.processing['parallel_threshold']
DISTANCE_THRESHOLD = config.processing['distance_threshold']
METERS_PER_PIXEL_X = config.processing['meters_per_pixel_x']
METERS_PER_PIXEL_Y = config.processing['meters_per_pixel_y']
YELLOW_LOWER = config.processing['yellow_lower']
YELLOW_UPPER = config.processing['yellow_upper']


def lane_mask(image, vertical_offset=0):
    window = image[vertical_offset:, :, :]
    yellow = extract_yellow(window)
    highlights = extract_highlights(window[:, :, 0])
    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)
    s_x = _sobel(gray, True, kernel_size=KERNEL_SIZE)
    s_y = _sobel(gray, False, kernel_size=KERNEL_SIZE)
    grad_dir, grad_mag = _gradient(s_x, s_y)
    mask = np.zeros(image.shape[:-1], dtype=np.uint8)
    mask[vertical_offset:, :][((s_x >= 25) & (s_x <= 255) &
                               (s_y >= 25) & (s_y <= 255)) |
                              ((grad_mag >= 30) & (grad_mag <= 512) &
                               (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                              (yellow == 255) |
                              (highlights == 255)] = 1
    return _filter_2d(mask, FILTER_2D_THRESHOLD)


def _sobel(image_ch, horizontal=True, kernel_size=3):
    return np.absolute(cv2.Sobel(image_ch, -1, *((1, 0) if horizontal else (0, 1)), ksize=kernel_size))


def _gradient(sobel_x, sobel_y):
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32), np.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.uint16)


def extract_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)


def extract_highlights(image, p=99.9):
    p = int(np.percentile(image, p) - 30)
    return cv2.inRange(image, p, 255)


def _filter_2d(image, thresh):
    k = np.array(FILTER_2D)
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < thresh] = 0
    return image


def lane_detection_histogram(image, steps, search_window, h_window):
    all_x, all_y = [], []
    masked_image = image[:, search_window[0]:search_window[1]]
    pixels_per_step = image.shape[0] // steps

    for i in range(steps):
        start = masked_image.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_image[end:start, :], axis=0)
        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))
        highest_peak = _highest_peaks(histogram_smooth, peaks, n=1, threshold=5)
        if len(highest_peak) == 1:
            highest_peak = highest_peak[0]
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_image, highest_peak, center, pixels_per_step)
            all_x.extend(x)
            all_y.extend(y)
    return np.array(all_x) + search_window[0], np.array(all_y)


def _highest_peaks(histogram, peaks, n=2, threshold=0):
    peaks_descending = sorted([(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold],
                              key=lambda _: _[1], reverse=True)
    return list(islice((peak for peak, _ in peaks_descending), n))


def get_pixel_in_window(image, x_center, y_center, size):
    size_half = int(size // 2)
    x_center = int(x_center)
    window = image[y_center - size_half:y_center + size_half, x_center - size_half:x_center + size_half]
    x, y = (window.T == 1).nonzero()
    return x + x_center - size_half, y + y_center - size_half


def remove_outliers(x, y, outlier_percentile=OUTLIER_PERCENTILE):
    if len(x) == 0 or len(y) == 0:
        return x, y

    x, y = np.array(x), np.array(y)
    lower_bound = np.percentile(x, outlier_percentile)
    upper_bound = np.percentile(x, 100 - outlier_percentile)
    criteria = (x >= lower_bound) & (x <= upper_bound)
    return x[criteria], y[criteria]


def acceptable_lanes(lane_one, lane_two, parallel_thresh=PARALLEL_THRESHOLD, distance_thresh=DISTANCE_THRESHOLD):
    is_parallel = lane_one.is_current_fit_parallel(lane_two, threshold=parallel_thresh)
    dist = lane_one.get_current_fit_distance(lane_two)
    is_plausible_dist = distance_thresh[0] < dist < distance_thresh[1]
    return is_parallel & is_plausible_dist


def curvature(curve, meters_per_pixel_x=METERS_PER_PIXEL_X, meters_per_pixel_y=METERS_PER_PIXEL_Y):
    y = np.array(np.linspace(0, 719, num=10))
    x = np.array([curve(x) for x in y])
    y_eval = np.max(y)
    curve = np.polyfit(y * meters_per_pixel_y, x * meters_per_pixel_x, 2)
    return ((1 + (2 * curve[0] * y_eval / 2. + curve[1]) ** 2) ** 1.5) / np.absolute(2 * curve[0])


def lane_detection_history(image, poly, steps):
    pixels_per_step = image.shape[0] // steps
    all_x, all_y = [], []
    for i in range(steps):
        start = image.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        center = (start + end) // 2
        x = poly(center)
        x, y = get_pixel_in_window(image, x, center, pixels_per_step)
        all_x.extend(x)
        all_y.extend(y)
    return all_x, all_y


def calculate_lane_area(lanes, area_height, steps):
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))
    for i in range(steps + 1):
        pixels_per_step = area_height // steps
        start = area_height - i * pixels_per_step
        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]
    return np.concatenate((points_left, points_right[::-1]), axis=0)


def draw_polynomial(image, polynomial, steps, color, thickness=10, dashed=False, tip_length=None):
    image_height = image.shape[0]
    pixels_per_step = image_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step
        start_point = (int(polynomial(start)), start)
        end_point = (int(polynomial(end)), end)
        if dashed is False or i % 2 == 1:
            if tip_length:  # draw arrowed line
                image = cv2.arrowedLine(image, end_point, start_point, color, thickness, tipLength=tip_length)
            else:
                image = cv2.line(image, end_point, start_point, color, thickness)
    return image

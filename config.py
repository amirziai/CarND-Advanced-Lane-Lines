import numpy as np

calibration = dict(
    image_size=(1280, 720),
    calibration_image_size=(720, 1280, 3),
    calibration_pickle_file='calibration.pkl',
    images_path='camera_cal/calibration*.jpg',
    chessboard_rows=6,
    chessboard_cols=9
)

processing = dict(
    kernel_size=3,
    filter_2d=[[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    filter_2d_threshold=4,
    histogram_peaks=2,
    outlier_percentile=5,
    parallel_threshold=(0.0003, 0.55),
    distance_threshold=(350, 460),
    meters_per_pixel_x=3.7 / 700,
    meters_per_pixel_y=30 / 720,
    yellow_lower=(20, 50, 150),
    yellow_upper=(40, 255, 255)
)

lane_detection = dict(
    arrow_tip_length=0.5,
    vertical_offset=400,
    histogram_window=7,
    polynomial_coefficient=719,
    line_segments=10
)

_annotate = dict(
    offset=250,
    src=np.float32([(132, 703), (540, 466), (740, 466), (1147, 703)])
)


annotate = dict(
    offset=_annotate['offset'],
    src=_annotate['src'],
    history=7,
    dst=np.float32([(_annotate['src'][0][0] + _annotate['offset'], 720),
                    (_annotate['src'][0][0] + _annotate['offset'], 0),
                    (_annotate['src'][-1][0] - _annotate['offset'], 0),
                    (_annotate['src'][-1][0] - _annotate['offset'], 720)]),
    annotated_video_suffix='_annotated.mp4'
)


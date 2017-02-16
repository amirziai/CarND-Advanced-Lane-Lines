import config

from scipy.misc import imread
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt

from calibration import Calibration
from lane_detection import LaneDetector

SRC = config.annotate['src']
DST = config.annotate['dst']
OFFSET = config.annotate['offset']
HISTORY = config.annotate['history']
ANNOTATED_VIDEO_SUFFIX = config.annotate['annotated_video_suffix']

calibration = Calibration()
lane_detector = LaneDetector(SRC, DST, n_images=HISTORY, calibration=calibration, offset=OFFSET)


def annotate_image(image_path):
    """Returns annotated image"""
    image = imread(image_path)
    image_annotated = lane_detector.process_image(image)
    return image_annotated


def annotate_image_plot(image_path, figure_size=(14, 10)):
    plt.figure(figsize=figure_size)
    plt.imshow(annotate_image(image_path));


def annotate_video(video_path, save=False):
    """Returns or saves annotated video"""
    video = VideoFileClip(video_path)
    video_annotated = video.fl_image(lane_detector.process_image)
    if save:
        video_annotated_file_name = '{}{}'.format(video_path.split('.')[0], ANNOTATED_VIDEO_SUFFIX)
        video_annotated.write_videofile(video_annotated_file_name, audio=False)
    else:
        return video_annotated

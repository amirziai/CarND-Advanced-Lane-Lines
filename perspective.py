import cv2
from functools import partial


class Perspective:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.m = cv2.getPerspectiveTransform(src, dst)
        self.m_inv = cv2.getPerspectiveTransform(dst, src)

    @staticmethod
    def _transform(image):
        return partial(cv2.warpPerspective, src=image, dsize=image.shape[:2][::-1], flags=cv2.INTER_LINEAR)

    def transform(self, image):
        return self._transform(image)(M=self.m)

    def inverse_transform(self, image):
        return self._transform(image)(M=self.m_inv)

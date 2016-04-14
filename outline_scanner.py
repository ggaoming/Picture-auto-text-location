# -*- coiding -*-
import cv2
from picture_loader import pictureReader

class outline_Scanner(object):

    def __init__(self, img_set):

        if img_set is None:
            img_set = pictureReader()
            img_set.load()
        self.img_set = img_set
        pass

    def scanner(self):
        pass

app = outline_Scanner(img_set=None)



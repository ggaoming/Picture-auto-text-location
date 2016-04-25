# -*-coding:utf8 -*-
import cv2
import numpy as np
from picture_loader import pictureReader


class object_scanner(object):

    def __init__(self, img_set=None, debug=False):
        if img_set is None:
            img_set = pictureReader()
            img_set.load()
        self.img_set = img_set
        self.debug = debug
        pass

    def resize(self, image, width=500):
        '''
        resize the image and convert to gray image
        :param image:
        :param width:
        :return: resize gray image
        '''
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (width, width))
        return gray_image
        pass

    def get_key_points(self, image, width=500):

        h, w, d = np.shape(image)
        if w != width or h != width or d != 1:
            image = self.resize(image=image, width=width)
        orb = cv2.ORB_create()
        kp = orb.detect(image)
        kp, des = orb.compute(img, kp)
        kp = kp[0:100]
        kp_pt = [x.pt for x in kp]
        kp_pt = np.array(kp_pt)

        if self.debug and len(kp_pt):
            print 'key points number %d' % (len(kp_pt))
            x = kp_pt[:, 0] - 50
            y = kp_pt[:, 1] - 50
            print ' ', np.sum(x), np.sum(y)
        return kp

    def scanner(self, image):
        kp = self.get_key_points(image)
        img = cv2.resize(image, (500, 500))
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255), flags=0)
        cv2.imshow('src', img2)
        cv2.waitKey()
        pass
'''
img = cv2.imread('1.png')
app = object_scanner(None, True)
for img in app.img_set.img_set:
    app.scanner(img)
'''
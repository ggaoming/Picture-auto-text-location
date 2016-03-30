# -*- coding:utf8 -*-
import cv2
import os
from picture_loader import pictureReader
import numpy as np
import skimage.color
from Palette import Palette
import sklearn.metrics


class imgHistProcess(object):

    def __init__(self, img_set, palette):

        if img_set is None:
            img_set = pictureReader()
            img_set.load()
        if palette is None:
            palette = Palette()

        self.img_set = img_set
        self.palette = palette
        pass

    def getHist(self):
        """

        :return:
        """
        pass

    def mainColor(self, img, debug = False):
        """
        get the main color of the image
        :param img:
        :return:
        """

        img_heigth, img_width, img_channels, = np.shape(img)
        if img_channels != 3:
            return None

        img_lab = skimage.color.rgb2lab(img)#cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab_array = img_lab.reshape((img_heigth * img_width, img_channels))

        dist = sklearn.metrics.euclidean_distances(img_lab_array, self.palette.lab_array)
        mindis_index = np.argmin(dist, axis=1)
        main_color_img = self.palette.lab_array[mindis_index, :]
        img = skimage.color.lab2rgb(main_color_img.reshape((img_heigth, img_width, img_channels)))
        cv2.imshow('src', img)
        cv2.waitKey()
        print mindis_index

        if debug:
            print 'img_label_array', img_lab_array
        return


app = imgHistProcess(img_set=None, palette=None)
app.mainColor(app.img_set.img_set[5], debug=True)


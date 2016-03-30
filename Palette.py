# -*-coding:utf8 -*-
import numpy as np
import skimage.color
import cv2

class Palette(object):
    """
    color palette
    """
    def __init__(self, num_hues=8, sat_range=2, light_range=2, debug=False):

        height = 1 + sat_range + (2 * light_range - 1)
        hues = np.tile(np.linspace(0, 1, num_hues + 1)[:-1], (height, 1))
        if num_hues == 8:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.51, 0.58, 0.77,  0.85]), (height, 1))
        if num_hues == 9:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.7, 0.87]), (height, 1))
        if num_hues == 10:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.66, 0.76, 0.87]), (height, 1))
        elif num_hues == 11:
            hues = np.tile(np.array(
                [0.0, 0.0833, 0.166, 0.25,
                 0.333, 0.5, 0.56333,
                 0.666, 0.73, 0.803,
                 0.916]), (height, 1))

        sats = np.hstack((
            np.linspace(0, 1, sat_range + 2)[1:-1],
            1,
            [1] * (light_range),
            [.4] * (light_range - 1),
        ))
        lights = np.hstack((
            [1] * sat_range,
            1,
            np.linspace(1, 0.2, light_range + 2)[1:-1],
            np.linspace(1, 0.2, light_range + 2)[1:-2],
        ))

        sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))
        lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))

        colors = skimage.color.hsv2rgb(np.dstack((hues, sats, lights)))
        grays = np.tile(
            np.linspace(1, 0, height)[:, np.newaxis, np.newaxis], (1, 1, 3))

        self.rgb_image = np.hstack((colors, grays))

        color_h, color_w, color_d = np.shape(colors)
        self.color_array = colors.reshape((color_h * color_w), color_d)
        if debug:
            cv2.imshow('src', self.rgb_image)
            cv2.waitKey()
        pass

    def save_img(self):

        pass

app = Palette(debug=True)
app.save_img()



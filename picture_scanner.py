cc# -*- coding:utf8 -*-
import cv2
import numpy as np

class Picture_Scanner(object):
    def __init__(self, src_img):
        """
        init of class
        :param src_img: source image 3 channels not gray color image
        :return:
        """
        self.img = src_img
        self.scan_step = 0.1

    def color_distances(self):
        """

        :return:
        """
        pass

    def object_weight_distances(self):
        """

        :return:
        """
        pass

    def focus_weight_distances(self):
        """

        :return:
        """
        pass

    def devide_into_patrs(self):
        """
        devide src img into parts
        :return:
        """
        img_shape = self.img.shape
        img_height = img_shape[0]
        img_width = img_shape[1]
        img_channels = img_shape[2]

        for y in range(int(1/self.scan_step)):
            for x in range(int(1/self.scan_step)):
                x0 = x * self.scan_step * img_width
                x1 = x0 + img_width*self.scan_step
                y0 = y * self.scan_step * img_height
                y1 = y0 + img_height*self.scan_step
                sub_img = self.img[int(y0):int(y1), int(x0):int(x1)]
                cv2.imshow('sub', sub_img)
                cv2.waitKey()

    def luminance_calculate(self):
        """
        calculate src image luminance information
        :return:
        """

        #convert rgb to hls use l channel

        hls_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)

        h_img, l_img, s_img = cv2.split(hls_img)

        sum_value = np.sum(l_img)
        u_map = l_img - sum_value/(np.shape(h_img)[0] * np.shape(h_img)[1])

        a_val = np.sqrt(np.sum((l_img - u_map) ** 2) / ((np.shape(h_img)[0] * np.shape(h_img)[1]) - 1))

        cv2.imshow('l', u_map/a_val)

        return u_map / a_val

img = cv2.imread('pictures/1.jpg')
app = Picture_Scanner(img)
app.luminance_calculate()
cv2.imshow('src', app.img)


cv2.waitKey()



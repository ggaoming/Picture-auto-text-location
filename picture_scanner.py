# -*- coding:utf8 -*-
import cv2
import numpy as np
import Utils
class Picture_Scanner(object):
    def __init__(self, src_img):
        """
        init of class
        :param src_img: source image 3 channels not gray color image
        :return:
        """
        self.img = src_img
        self.scan_step = 0.1
        self.feature_blocks = []

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
        :return: center value and gauss model
        """
        img_shape = self.img.shape
        img_height = img_shape[0]
        img_width = img_shape[1]
        img_channels = img_shape[2]

        index_x = 0
        index_y = 0
        for y in range(int(1/self.scan_step)):
            self.feature_blocks.append([])
            index_x = 0
            for x in range(int(1/self.scan_step)):
                x0 = x * self.scan_step * img_width
                x1 = x0 + img_width*self.scan_step
                y0 = y * self.scan_step * img_height
                y1 = y0 + img_height*self.scan_step
                sub_img = self.img[int(y0):int(y1), int(x0):int(x1)]
                mu, std = self.luminance_calculate(sub_img=sub_img)
                self.feature_blocks[y].append((mu, std))
                index_x += 1
            index_y += 1
        for i in self.feature_blocks:
            print i

    def luminance_calculate(self, sub_img):
        """
        calculate src image luminance information
        :return:
        """

        #convert rgb to hls use l channel

        hls_img = cv2.cvtColor(sub_img, cv2.COLOR_RGB2HLS)

        h_img, l_img, s_img = cv2.split(hls_img)

        sum_value = np.sum(l_img)
        u_map = l_img - sum_value/(np.shape(h_img)[0] * np.shape(h_img)[1])

        a_val = np.sqrt(np.sum((l_img - u_map) ** 2) / ((np.shape(h_img)[0] * np.shape(h_img)[1]) - 1))


        x_array = []
        for j in range(np.shape(u_map)[0]):
            for i in range(np.shape(u_map)[1]):
                x_array.append(u_map[j, i])
        mu, std = Utils.gauss_fit(x=x_array)
        return mu, std



img = cv2.imread('pictures/1.jpg')
app = Picture_Scanner(img)
app.devide_into_patrs()
cv2.imshow('src', app.img)


cv2.waitKey()



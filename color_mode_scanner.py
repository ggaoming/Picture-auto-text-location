# -*- coding:utf8 -*-
import cv2
import os
from picture_loader import pictureReader
import numpy as np
import skimage.color
from Palette import Palette
import sklearn.metrics
from sklearn.metrics.pairwise import  manhattan_distances


class imgHistProcess(object):

    def __init__(self, img_set, palette, debug):

        if palette is None:
            palette = Palette()
        self.palette = palette

        if img_set is None:
            img_set = pictureReader()
            img_set.load()
        if img_set.img_set_hist is None or len(img_set.img_set_hist)<0:
            img_set.img_set_hist = self.get_set_Hist(img_set.img_set)

        self.img_set = img_set

        self.debug = debug
        self.img_set_hists = []
        if debug:
            print 'init '
        pass

    def get_set_Hist(self, img_set):
        """
        :param img:
        :return:
        """
        color_hist_set = []
        for img in img_set:
            img_heigth, img_width, img_channels, = np.shape(img)
            if img_channels != 3:
                return None
            num_colors = self.palette.lab_array.shape[0]
            num_pixels = img_heigth*img_width
            img_lab = skimage.color.rgb2lab(img)#cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img_lab_array = img_lab.reshape((img_heigth * img_width, img_channels))
            dist = sklearn.metrics.euclidean_distances(img_lab_array, self.palette.lab_array)
            min_ind = np.argmin(dist, axis=1)  # the index of the palette's colors
            color_hist = 1.0 * np.bincount(min_ind, minlength=num_colors) / num_pixels # create the hist
            color_hist_set.append(color_hist)
        return color_hist_set

    def getHist(self, img, img_lab_array):
        """
        create img's hist according to the img's lab array
        img (w,h,3)
        img_lab_array( w*h , 3) rgb2lab
        :return:
        """
        if self.debug:
            print 'plate array shape: l,d :', self.palette.lab_array.shape[0]  # palette's shape[0] length
            print 'pixs num: ', img_lab_array.shape[0]  # image's pixs numbers
        num_colors = self.palette.lab_array.shape[0]
        num_pixels = img_lab_array.shape[0]
        dist = sklearn.metrics.euclidean_distances(self.palette.lab_array, img_lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)  # the index of the palette's colors
        color_hist = 1.0 * np.bincount(min_ind, minlength=num_colors) / num_pixels # create the hist
        """
        np.bincount()
            get the number of the elements
            example:
                a = np.array([1,1,1,2,2])
                np.bincount(a)
                array([0, 3, 2])
        """
        if self.debug:
            print self.palette.lab_array[0]
            print 'color hist', color_hist
        return color_hist
        pass

    def get_nearst_n_hist_index(self, color_hist, num=10):
        """
        return n closest nearest to the color_hist and the distance to it
        :param color_hist: hist of the color palette
        :param num: n of the nearest to return
        :return:
        nearst_index: the index of the neighbors
        nearst_dists: the distance to the neighbors
        """
        #------ manhattan_distance ------
        #------ 曼哈顿距离
        #------ 直方图相似度评估
        dists = manhattan_distances(color_hist, self.img_set.img_set_hist)
        dists = dists.flatten()
        nearst_index = np.argsort(dists).flatten()[:num]
        nearst_dists = dists[nearst_index]
        nearst_file_name = []
        for i in nearst_index:
            nearst_file_name.append(self.img_set.img_load_order[i])
        if self.debug:

            print 'pic file names', nearst_file_name
        return nearst_index, nearst_dists

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

        if self.debug:
            print mindis_index
            print 'img_label_array', img_lab_array
            pass
        return img, main_color_img

img = cv2.imread('1.png');
app = imgHistProcess(img_set=None, palette=None, debug=True)
palette_img, img_main_color_array = app.mainColor(img)
hist = app.getHist(palette_img, img_main_color_array)
app.get_nearst_n_hist_index(color_hist=hist, num=10)



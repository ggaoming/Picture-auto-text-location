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
        pass


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

def test0():
    j = 1
    img = cv2.imread('pictures/%d.jpg' % j)
    app = Picture_Scanner(img)
    app.devide_into_patrs()
    min_value = np.inf
    min_index = 0
    for i in range(42):
        if i == j:
            continue
        img2 = cv2.imread('pictures/%d.jpg' % (i))
        app2 = Picture_Scanner(img2)
        app2.devide_into_patrs()
        arr, val = Utils.img1_img2_compare_mu_std(app.feature_blocks, app2.feature_blocks)
        print i, ":", arr, val
        if val < min_value:
            min_index = i
            min_value = val
    print min_index, min_value

    cv2.waitKey()
def test2():

    X = []
    for i in range(20):
        img2 = cv2.imread('pictures/%d.jpg' % (i))
        app2 = Picture_Scanner(img2)
        app2.devide_into_patrs()
        """
        a = []
        for i in range(len(app2.feature_blocks)):
            for j in range(len(app2.feature_blocks[0])):
                a.append(np.abs(app2.feature_blocks[i][j]))
        X.append(a)
        """
        X.append(app2.feature_blocks)
    result = Utils.hcluster_func(X=X[1:], target=X[0], group_size=5)

test2()




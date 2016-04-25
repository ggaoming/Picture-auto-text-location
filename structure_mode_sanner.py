# -*- coding:utf8 -*-
'''
计算图像的结构相似行
'''
from picture_loader import pictureReader
import numpy as np
import cv2
from scipy.ndimage.filters import correlate1d


class Structure_Process():

    def __init__(self, img_set=None, debug=False):

        if img_set is None:
            img_set = pictureReader()
            img_set.load()
        self.img_set = img_set
        self.block_shape = (5, 5)
        self.debug = debug
        self.img_sub_set = None
        '''
        开辟内存过大　此处删去
        if self.img_set.img_set_mu_std is None:
            img_set.img_set_mu_std = self.process_set_data()
            self.img_set.img_set_mu_std = img_set.img_set_mu_std
        '''
        print '*** structure init done ***'
        pass

    def process_set_data(self):
        '''
        devide the img_set's img into parts
        :return:
        '''
        sub_img_set = []
        img_value_set = []
        for img in self.img_set.img_set:
            sub_imgs, values = self.devide_into_parts(img=img)
            sub_img_set.append(sub_imgs)
            img_value_set.append(values)
        if self.debug:
            print 'img set process done'
        self.img_sub_set = sub_imgs
        return img_value_set

    def devide_into_parts(self, img):
        '''
        devide a img into parts
        shape by self.block_shape
        :param img: input image np array
        :return: sub images
        '''
        h, w, d = np.shape(img)
        if self.debug:
            print 'devide img shape', h, w, d
        sub_h = int(h/self.block_shape[0])
        sub_w = int(w/self.block_shape[1])
        index_x = 0
        index_y = 0
        img_sub_set = []
        img_sub_set_values = []
        for y in range(self.block_shape[0]):
            img_sub_set.append([])
            img_sub_set_values.append([])
            for x in range(self.block_shape[1]):
                s_y = sub_h*y
                s_x = sub_w*x
                e_x = s_x + sub_w
                e_y = s_y + sub_h
                sub_img = img[s_y:e_y, s_x:e_x]
                mu, std = self.calculate_mu_std(img=sub_img)
                #test img_sommth, img_org = self.gaussian_bur(img=sub_img)
                img_sommth, img_org = self.gaussian_bur2(img=sub_img)
                img_sub_set[index_y].append((img_org, img_sommth))
                img_sub_set_values[index_y].append((mu, std))
                pass
            index_y += 1
            pass
        return img_sub_set, img_sub_set_values
        pass

    def calculate_mu_std(self, img):
        return None, None

    def create_gaussian_kernel(self, width=11, sigma=1.5):
        kernel = np.ndarray((width))
        norm_mu = int(width/2)
        for i in range(width):
            kernel[i] = (1/np.sqrt(2*np.pi)*sigma)*np.exp(-((i-norm_mu)**2)/(2*(sigma**2)))
        return kernel
        pass
    def gaussian_bur_utils(self, img, kernel=None):
        if kernel is None:
            kernel = self.create_gaussian_kernel()
        temp = np.asarray(img).astype(np.float)
        temp = correlate1d(temp, kernel, axis=0)
        temp = correlate1d(temp, kernel, axis=1)
        return temp
        pass

    def gaussian_bur2(self, img):
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h_img, l_img, s_img = cv2.split(hls_img)
        h_sommth_img = self.gaussian_bur_utils(h_img)
        l_sommth_img = self.gaussian_bur_utils(l_img)
        s_sommth_img = self.gaussian_bur_utils(s_img)
        return (h_sommth_img, l_sommth_img, s_sommth_img), (h_img, l_img, s_img)
        pass


    def gaussian_bur(self, img):
        '''
        use opencv to apply guassian bur on the img
        :param img:
        :return: h channel's smoothed img
        '''
        width = 11
        sigma = 1.5
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h_img, l_img, s_img = cv2.split(hls_img)
        h_sommth_img = cv2.GaussianBlur(h_img, (width, width), sigma)
        l_sommth_img = cv2.GaussianBlur(l_img, (width, width), sigma)
        s_sommth_img = cv2.GaussianBlur(s_img, (width, width), sigma)
        return (h_sommth_img, l_sommth_img, s_sommth_img), (h_img, l_img, s_img)
        pass


    def calculate_distance(self, sub_set0, sub_set1):
        box_h, box_w, t, c, sub_h, sub_w = np.shape(sub_set0)
        SSIM = []
        for sub_box_h in range(box_h):
            for sub_box_w in range(box_w):
                #the org img is split into box_h*box_w parts
                org_sub_img0 = sub_set0[sub_box_h][sub_box_w][0]  # (h_img, l_img, s_img)
                smooth_sub_img0 = sub_set0[sub_box_h][sub_box_w][1]  # (h_sommth_img, l_sommth_img, s_sommth_img)
                org_sub_img1 = sub_set1[sub_box_h][sub_box_w][0]
                smooth_sub_img1 = sub_set1[sub_box_h][sub_box_w][1]
                _, _, gray0 = org_sub_img0
                _, _, gray0_mu = smooth_sub_img0
                _, _, gray1 = org_sub_img1
                _, _, gray1_mu = smooth_sub_img1
                _, gray0, _ = org_sub_img0
                _, gray0_mu, _ = smooth_sub_img0
                _, gray1, _ = org_sub_img1
                _, gray1_mu, _ = smooth_sub_img1

                gray0_squared = gray0 ** 2
                gray0_mu_squared = gray0_mu ** 2

                #test gray0_sigma_squared = cv2.GaussianBlur(gray0_squared, (11, 11), 1.5) - gray0_mu_squared
                gray0_sigma_squared = self.gaussian_bur_utils(gray0_squared) - gray0_mu_squared

                gray1_squared = gray1 ** 2
                gray1_mu_squared = gray1_mu ** 2

                #test gray1_sigma_squared = cv2.GaussianBlur(gray1_squared, (11, 11), 1.5) - gray1_mu_squared
                gray1_sigma_squared = self.gaussian_bur_utils(gray1_squared) - gray1_mu_squared

                mat_12 = gray0 * gray1
                #mat_sigma_12 = cv2.GaussianBlur(mat_12, (11, 11), 1.5)
                mat_sigma_12 = self.gaussian_bur_utils(mat_12)
                mat_mu_12 = gray0_mu * gray1_mu
                mat_sigma_12 -= mat_12

                c1 = (0.01 * 255)**2
                c2 = (0.03 * 255)**2
                num_ssim = (2 * mat_mu_12 + c1) * (2 * mat_sigma_12 + c2)
                den_ssim = (gray0_mu + gray1_mu_squared + c1)*(gray0_sigma_squared + gray1_sigma_squared + c2)
                sub_ssim = num_ssim/den_ssim
                print sub_ssim
                sub_ssim = np.average(sub_ssim)
                print sub_ssim
                return
        SSIM = np.array(SSIM)
        SSIM = np.average(SSIM)
        if self.debug:
            print 'result:', np.average(SSIM)
        return SSIM
        pass

    def compare_img(self, img):
        '''
        compare with the img_set with the input img
        :param img: input image
        :return: compare result
        '''
        img_index = 0
        SSIM_list = []
        for target_img in self.img_set.img_set:
            h, w, _ = np.shape(target_img)
            cv2.imshow('1',target_img)
            cv2.imshow('2',img)
            temp_img = cv2.resize(img, (w, h))
            sub_imgs_set0, _ = self.devide_into_parts(temp_img)
            sub_imgs_set1, _ = self.devide_into_parts(target_img)
            ssim = self.calculate_distance(sub_imgs_set0,sub_imgs_set1)
            SSIM_list.append(ssim)
            cv2.waitKey()
            return
            img_index += 1

        return None

img = cv2.imread('1.png')
app = Structure_Process(debug=False)
app.compare_img(img=app.img_set.img_set[0])

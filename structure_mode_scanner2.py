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
        '''
        开辟内存过大　此处删去
        if self.img_set.img_set_mu_std is None:
            img_set.img_set_mu_std = self.process_set_data()
            self.img_set.img_set_mu_std = img_set.img_set_mu_std
        '''
        print '*** structure init done ***'
        pass

    def create_gaussian_kernel(self, width=11, sigma=1.5):
        kernel = np.ndarray(width)
        norm_mu = int(width/2)
        for i in range(width):
            kernel[i] = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((i - norm_mu) ** 2) / (2 * (sigma ** 2))))
        return kernel

    def gauss_bur_util(self, image, kernel=None):
        if kernel is None:
            kernel = self.create_gaussian_kernel()
        result = correlate1d(image, kernel, axis=0)
        result = correlate1d(result, kernel, axis=1)
        return result

    def convert_to_signel_channel(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray).astype(np.float)
        return gray
        pass

    def compute_ssim_gauss(self, image1, image2):

        image1_gray = self.convert_to_signel_channel(image=image1)
        image1_gray_squared = image1_gray ** 2
        image1_gray_mu = self.gauss_bur_util(image1_gray)
        image1_gray_mu_squared = image1_gray_mu ** 2
        image1_gray_sigma_squared = self.gauss_bur_util(image1_gray_squared)
        image1_gray_squared -= image1_gray_squared

        image2_gray = self.convert_to_signel_channel(image=image2)
        image2_gray_squared = image2_gray ** 2
        image2_gray_mu = self.gauss_bur_util(image2_gray)
        image2_gray_mu_squared = image2_gray_mu ** 2
        image2_gray_sigma_squared = self.gauss_bur_util(image2_gray_squared)
        image2_gray_squared -= image2_gray_squared

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2


        image_mat_12 = image1_gray * image2_gray
        image_mat_sigma_12 = self.gauss_bur_util(image_mat_12)
        image_mat_mu_12 = image1_gray_mu * image2_gray_mu
        image_mat_sigma_12 = image_mat_sigma_12 - image_mat_mu_12

        num_ssim = (2 * image_mat_mu_12 + c1) * (2 * image_mat_sigma_12 + c2)

        den_ssim = (image1_gray_mu_squared + image2_gray_mu_squared + c1)\
                   * (image1_gray_sigma_squared + image2_gray_sigma_squared + c2)

        ssim_map = num_ssim /  den_ssim
        index = np.average(ssim_map)
        #print ssim_map
        print index
        print '%.7g'%(index)

    def compute_ssim(self, image1, image2):
        '''
        use 11*11 block to scan the image1
        calculate ssim value map
        use the mean value as the ssim result
        :param image1: input image1
        :param image2: input image2
        :return:
        '''
        c1 = 6.5025
        c2 = 58.5225

        I1 = np.asarray(image1).astype(np.float32)
        I2 = np.asarray(image2).astype(np.float32)

        I2_2 = I2 ** 2
        I1_2 = I1 ** 2
        I1_I2 = I1 * I2

        mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

        mu1_2 = mu1*mu1
        mu2_2 = mu2*mu2
        mu1_mu2 = mu1*mu2

        sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5) - mu1_2
        sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5) - mu2_2
        sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5) - mu1_mu2


        t1 = 2 * mu1_mu2 + c1
        t2 = 2 * sigma12 + c2
        t3 = t1 * t2

        t1 = mu1_2 + mu2_2 + c1
        t2 = sigma1_2 + sigma2_2 + c2
        t1 = t1 * t2

        ssim_map = t3 / t1
        mssim = np.mean(ssim_map)
        if self.debug:
            print "mean %.7g; avg %.7g"%(np.mean(ssim_map), np.average(ssim_map))
        return np.mean(ssim_map)




    def compare_img(self, img):
        for target_img in self.img_set.img_set:
            h, w, _ = np.shape(target_img)
            temp_img = cv2.resize(img, (w, h))
            self.compute_ssim(target_img, temp_img)
            cv2.imshow('1',self.gauss_bur_util(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)))
            cv2.imshow('2',self.gauss_bur_util(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)))
            cv2.waitKey()
            #return

            pass
img = cv2.imread('1.png')
app = Structure_Process(debug=True)
app.compare_img(app.img_set.img_set[0])

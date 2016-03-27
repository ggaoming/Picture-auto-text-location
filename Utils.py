# -*- coding:utf8 -*-
import scipy
import numpy as np
from scipy.stats import norm

def gauss_fit(x):
    """
    :param x:
    :return:
    """
    mu, std = norm.fit(x)

    return mu, std
    pass

def gauss_verification(mu, std, x):
    """
    :param mu:
    :param std:
    :param x:
    :return:
    """
    pass

def img1_img2_compare_mu_std(img1_feature, img2_feature):

    if len(img1_feature) != len(img2_feature) or len(img2_feature[0]) != len(img2_feature[0]):
        return None
    a = []
    b = []
    for i in range(len(img1_feature)):
        for j in range(len(img1_feature[0])):
            a.append(np.abs(img1_feature[i][j])+1.0)
            b.append(np.abs(img2_feature[i][j])+1.0)

    a = np.array(a)
    b = np.array(b)
    a_sum = np.sum(a)
    b_sum = np.sum(b)

    #use bhattacharyya to measure similarity
    temp = 0.0
    for i in range(len(a)):
        temp += np.sqrt(a[i]*b[i])/(a_sum * b_sum)
    bhattacharyya_distance = np.sqrt(1 - temp)

    return bhattacharyya_distance, np.sqrt(np.sum(bhattacharyya_distance**2))

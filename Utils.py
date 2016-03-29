# -*- coding:utf8 -*-
import scipy
import numpy as np
from scipy.stats import norm
import scipy.cluster.hierarchy as hcluster

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
    """
    :param img1_feature: image 1's feature
    :param img2_feature: image 2's feature
    :return:
    """

    if len(img1_feature) != len(img2_feature) or len(img2_feature[0]) != len(img2_feature[0]):
        return None
    a = []
    b = []
    for i in range(len(img1_feature)):
        for j in range(len(img1_feature[0])):
            a.append(np.abs(img1_feature[i][j]) + 1.0)
            b.append(np.abs(img2_feature[i][j]) + 1.0)
    a = np.array(a)
    b = np.array(b)
    a_sum = np.sum(a)
    b_sum = np.sum(b)

    #use bhattacharyya to measure similarity
    temp = 0.0
    for i in range(len(a)):
        temp += np.sqrt(a[i]*b[i])/(a_sum * b_sum)
    bhattacharyya_distance = np.sqrt(1 - temp)

    return bhattacharyya_distance, np.sqrt(np.sum((a_sum - b_sum)**2))


def hcluster_func(X, target, group_size):
    import copy
    loop_depth = 5
    target_loop = [target]
    train_set =copy.copy(X)
    for LOOP in range(loop_depth):
        print '***loop***', LOOP
        minIndex = -1
        minDistance = np.inf
        for ele0 in train_set:
            distance = []
            for ele1 in target_loop:
                distance0, distance1 = img1_img2_compare_mu_std(ele0, ele1)
                distance.append(distance1)
            currValue = sum(distance)/len(distance)
            if minDistance > currValue:
                minDistance = currValue
                minIndex = train_set.index(ele0)
        print ' ', 'minDistance %lf minValue %d' % (minDistance, minIndex)
        currArray = train_set[minIndex]
        train_set.remove(currArray)
        target_loop.append(currArray)


    index = 0
    for ele in target_loop:
        if ele in X:
            index += 1
            print "Index %d train_Id: %d" % (index, X.index(ele))
    pass

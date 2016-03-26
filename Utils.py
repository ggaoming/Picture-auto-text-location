# -*- coding:utf8 -*-
import scipy
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
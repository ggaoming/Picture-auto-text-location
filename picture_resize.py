import cv2
import os
import numpy as np

FILEPATH = 'pictures/'
picture_name = os.listdir(FILEPATH)
picture_path = [FILEPATH + f for f in picture_name]
for f in picture_path:
    if not '.jpg' in f:
        continue
    filename = os.path.split(f)[-1]
    img = cv2.imread(f)
    h, w, d = np.shape(img)
    if h > 500:
        d_h = int(500)
        d_w = int(500 * w / h)
        h = d_h
        w = d_w
    if w > 500:
        d_w = int(500)
        d_h = int(500 * h / w)
        w = d_w
        h = d_h
    new_img = cv2.resize(img, (w, h))
    cv2.imshow('n', new_img)
    os.remove(f)
    cv2.imwrite(f, new_img)
    print h, w
    cv2.waitKey()

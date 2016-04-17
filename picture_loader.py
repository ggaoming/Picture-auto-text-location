# -*- coding:utf8 -*-
"""
load pictures
"""
import os
import cv2
import json
FILEPATH = 'pictures/'
class pictureReader(object):

    def __init__(self):
        self.img_set = []  # 保存图像数据
        self.img_load_order = []  # 保存数据载入顺序　文件名
        self.text_area_set = []  # 文字区域
        self.text_area_dic = {}  # 文字区域
        #--- 区域选择使用参数　----#
        self.curr_start_x = -1
        self.curr_start_y = -1
        self.curr_end_x = -1
        self.curr_end_y = -1
        self.drag_start = 0
        #------******------#
        self.img_set_hist = None  # 图像颜色直方图数据
        '''
        value struct np.array() normalize into 1 weight
        order: in imgs' load order
        '''

        #------******------#
        self.img_set_mu_std = None  # 图像颜色结构方差和标准差
        '''
        value struct
            element( (mu_h, mu_l, mu_s), (std_h, std_l, std_s) ) int HLS color space
        order: in img split style
            box0 box1 box2
            box3 box4 box5
            box6 box7 box8
        '''
        pass

    def load(self):
        """
        get picture name in path FILEPATH
        载入图片
        :return: None
        """
        picture_name = os.listdir(FILEPATH)
        picture_path = [FILEPATH + f for f in picture_name]
        for f in picture_path:
            if not '.jpg' in f:
                continue
            filename = os.path.split(f)[-1]
            img = cv2.imread(f)
            self.img_set.append(img)
            self.img_load_order.append(filename)
            self.text_area_set.append([])
        print 'Image number :', len(picture_path)

    def get_text_areas(self):
        """
        figure out the text area
        手动选取文字区域
        :return:
        """
        if len(self.img_set) == 0:
            print 'no img array'
            return
        cv2.namedWindow('src')
        cv2.setMouseCallback('src', self.onmouse)
        image_index = 0
        for img in self.img_set:
            print self.img_load_order[image_index]
            self.curr_start_x = -1
            self.curr_start_y = -1
            self.curr_end_x = -1
            self.curr_end_y = -1

            while True:
                img_copy = img.copy()
                if self.drag_start == 1:
                    cv2.rectangle(img_copy, (self.curr_start_x, self.curr_start_y),
                                  (self.curr_end_x, self.curr_end_y),
                                  (255, 0, 0), 1)
                    pass
                elif self.drag_start == 2:
                    cv2.rectangle(img_copy, (self.curr_start_x, self.curr_start_y),
                                  (self.curr_end_x, self.curr_end_y),
                                  (0, 255, 0), 1)
                    pass
                if len(self.text_area_set[image_index]) > 0:
                    for n in self.text_area_set[image_index]:
                        cv2.rectangle(img_copy, (n[0], n[1]),
                                  (n[2], n[3]),
                                  (0, 0, 255), 1)
                cv2.imshow('src', img_copy)
                key_in = cv2.waitKey(10)
                if key_in == -1:
                    continue
                elif key_in == 27:
                    break
                elif key_in == ord('c'):
                    self.text_area_set[image_index] = []
                    continue
                elif key_in == ord('s'):
                    self.text_area_dic[self.img_load_order[image_index]] = self.text_area_set[image_index]
                    print self.text_area_dic
                    break
                elif key_in == ord('a'):
                    print 'add rect x0:%d y0:%d x1:%d y1%d' % (self.curr_start_x, self.curr_start_y,
                                                             self.curr_end_x, self.curr_end_y)
                    text_area = (self.curr_start_x, self.curr_start_y,
                                 self.curr_end_x, self.curr_end_y)
                    self.text_area_set[image_index].append(text_area)
            image_index += 1
        info = json.dumps(self.text_area_dic)
        file_w = open('image_info.json', 'w')
        file_w.write(info)
        file_w.close()

        pass


    def onmouse(self, event, x, y, flags, param):
        """
        opencv mouse function to draw area
        opencv函数 框选文字区域
        :param event:  event of mouse click
        :param x: mouse location.x
        :param y: mouse location.y
        :param flags:
        :param param:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.curr_start_x = x
            self.curr_start_y = y
            self.curr_end_x = x
            self.curr_end_y = y
            self.drag_start = 1
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_start = 2
            self.curr_end_x = x
            self.curr_end_y = y
            pass
        elif self.drag_start and self.drag_start == 1:
            self.curr_end_x = x
            self.curr_end_y = y
            pass
#app = pictureReader()





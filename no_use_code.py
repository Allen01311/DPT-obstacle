import os
import glob
import torch
import cv2
import argparse
import math
import util.io
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from util.misc import visualize_attention

#最大輪廓在總面積之占比
        #contour_area_ratio = area / total_area
        #print('Largest Contour Area Ratio : {:.2f}'.format(contour_area_ratio))

#最大外接矩形(只能正矩形)
        # x,y:外接矩形左上角座標；w:寬度；h:高度

image_path = "D:/image_experience/DPT-main/output_monodepth/org_559adb4438365561_1702107156000.png"
font = cv2.FONT_HERSHEY_SIMPLEX
image = cv2.imread(image_path)
resize_image = cv2.resize(image, (600, 400))
img_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)

#儲存gray_image
    # gray_output_path = os.path.join(output_folder_path, f"gray_{os.path.basename(image_path)}")
    # cv2.imwrite(gray_output_path, thresh)
    
# enhanced_image = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0)
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(11, 11))
clahe_image = clahe.apply(img_gray)
equalized_image = cv2.equalizeHist(clahe_image)
_, thresh = cv2.threshold(equalized_image, 170, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea, default=None)

if largest_contour is not None:
    total_area = resize_image.shape[0] * resize_image.shape[1]  #圖像總面積
    
    #最大輪廓面積
    area = cv2.contourArea(largest_contour)
    print(' Largest Contour Area :', area)
    print('---------------------------------------------')
    #-----------------------------------------------------------------------
    
    # 計算整張影像的中心點
    image_center_point = (int(resize_image.shape[1] / 2), int(resize_image.shape[0] / 2))
    print(' Image Center Point:', image_center_point)
    print('---------------------------------------------')
    #-----------------------------------------------------------------------
    
    # 最大外接矩形(可轉向)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx_polygon)
    print("Bounding Rect: (x={}, y={}, w={}, h={})".format(x, y, w, h))
    
    #最大外接矩形面積
    Bounding_Rect_Area = w * h
    print("Largest Bounding Rect Area : ", Bounding_Rect_Area)
    
    # #最大外接矩形面積之占比
    # bounding_area_ratio = Bounding_Rect_Area / total_area
    # print("Largest Bounding Area Ratio : {:.2f}".format(bounding_area_ratio))
    
    cv2.rectangle(resize_image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
    cv2.drawContours(resize_image, [largest_contour], -1, (0, 255, 0), 2)   
    cv2.drawContours(resize_image, [approx_polygon], -1, (0, 255, 255), 2)
    cv2.circle(resize_image, image_center_point, 5, (0, 0, 255), -1)
    output_path = 'D:/image_experience/DPT-main/error.jpg'
    cv2.imwrite(output_path, resize_image)  
#計算中心點到四邊的最短距離
        # upper_left = np.array([x,y])
        # upper_right = np.array([x+w,y])
        # lower_left = np.array([x,y+h])
        # lower_right = np.array([x+w,y+h])
        # top_distance = point_distance_line(image_center_point, upper_left, upper_right)
        # bottom_distance = point_distance_line(image_center_point, lower_left, lower_right)
        # left_distance = point_distance_line(image_center_point, lower_left, upper_left)
        # right_distance = point_distance_line(image_center_point, upper_right, lower_right)
        
#-------------------------------------------------------------------------------------------------
# #計算四個邊的中點
        # top_mid_point = (int(x + w / 2), y)
        # bottom_mid_point = (int(x + w / 2), y + h)
        # left_mid_point = (x, int(y + h / 2))
        # right_mid_point = (x + w, int(y + h / 2))

        # print("Top Mid Point :", top_mid_point)
        # print("Bottom Mid Point :", bottom_mid_point)
        # print("Left Mid Point :", left_mid_point)
        # print("Right Mid Point :", right_mid_point)
        
        # # 計算中心點至矩形各邊中點的距離
        # distance_to_top = math.dist(image_center_point, top_mid_point)
        # distance_to_bottom = math.dist(image_center_point, bottom_mid_point)
        # distance_to_left = math.dist(image_center_point, left_mid_point)
        # distance_to_right = math.dist(image_center_point, right_mid_point)

        # print("Distance to Top:", distance_to_top)
        # print("Distance to Bottom:", distance_to_bottom)
        # print("Distance to Left:", distance_to_left)
        # print("Distance to Right:", distance_to_right)
        
#-------------------------------------------------------------------------------------------------
# 繪製四條線段
        # cv2.line(resize_image, tuple(upper_left), tuple(upper_right), (0, 255, 0), 2)  # Top line
        # cv2.line(resize_image, tuple(lower_left), tuple(lower_right), (0, 255, 0), 2)  # Bottom line
        # cv2.line(resize_image, tuple(lower_left), tuple(upper_left), (0, 255, 0), 2)  # Left line
        # cv2.line(resize_image, tuple(upper_right), tuple(lower_right), (0, 255, 0), 2)  # Right line

        # 繪製各線段中點
        # cv2.circle(resize_image, top_mid_point, 5, (255, 0, 0), -1)
        # cv2.circle(resize_image, bottom_mid_point, 5, (255, 0, 0), -1)
        # cv2.circle(resize_image, left_mid_point, 5, (255, 0, 0), -1)
        # cv2.circle(resize_image, right_mid_point, 5, (255, 0, 0), -1)
        
        # 繪製GPS點至各線段中點距離
        # cv2.line(resize_image, image_center_point, top_mid_point, (200, 100, 200), 2)
        # cv2.line(resize_image, image_center_point, bottom_mid_point, (200, 100, 200), 2)
        # cv2.line(resize_image, image_center_point, left_mid_point, (200, 100, 200), 2)
        # cv2.line(resize_image, image_center_point, right_mid_point, (200, 100, 200), 2)
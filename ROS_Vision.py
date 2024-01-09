# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:25:21 2023

@author: user
"""

#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import apriltag
import cv2
import glob
import math
import numpy as np
import os
import rospy
import time

from bspline import *
#from bebop_msgs.msg import Ardrone3PilotingStateAltitudeChanged, Ardrone3PilotingStatePositionChanged, Ardrone3PilotingStateAttitudeChanged
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from distutils.util import strtobool
from numpy.linalg import inv
from pprint import pprint
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray,Header,Empty, String, Bool, Float64MultiArray
from skimage.measure import compare_ssim

import util.io
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from util.misc import visualize_attention
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import torch
import math
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

# show image for debug
show_image_flag = True
show_mask_flag = True


optimize = True
kitti_crop=False
absolute_depth=False
# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

net_w = net_h = 384
model = DPTDepthModel(
        path="weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
)
normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = Compose(
    [
        Resize
        (
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
)

model.eval()

if optimize == True and device == torch.device("cuda"):
  model = model.to(memory_format=torch.channels_last)
  model = model.half()

model.to(device)

class collect_data:
  
    def __init__(self):
    
      # image size
      self.width = 856  #856
      self.height = 480 #480
      self.img_cx = 428
      self.img_cy = 360
      ###############################################################################
    
      # setting
      self.usingUAV = "DJI"     # or "Bebop"
      self.usingRotView = False # or True, using RotView
      self.usingVTS = True      # or False, using LTS
      self.lookCenter = True    # or False, look top line 
      self.usingGrid = "rec"    # or "rec", using rectangle grid
      self.hsv_detect = False          # Hsv detected?
      self.apt_detect = False          # Apriltag detected?
      self.Detected_Stage = ""
      self.Target_Stage = ""
      self.HSVSize = 300                   # 紀錄面積大小用於除錯
      self.Timage = cv2.imread("005.png") # 匹配標的
      self.FirstTrack = True # 0609
      self.FirstAPT = False
      
      # image converter
      self.bridge = CvBridge()
      self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
      self.orb = cv2.ORB_create()
      self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
      self.tracker = cv2.TrackerCSRT_create() 
      self.pos_tracker = cv2.TrackerCSRT_create() 
      self.initBB = None
      self.union_data = []
      self.guess_data = []
      self.start_detect_landmark = False
      
      self.start_take_image = False #231216
      ###############################################################################
      
      # ros sub, pub
      self.point_pub = rospy.Publisher("/target_point", Int16MultiArray, queue_size = 10)
      #self.pub = rospy.Publisher("/dji/image", Image, queue_size=1)
      self.sub_start_detect_road = rospy.Subscriber("/control_landmark", Bool, self.start_detect_landmark_cb, queue_size=10)
      self.sub_start_take_image = rospy.Subscriber("/active_image", Bool, self.start_take_image_cb, queue_size=10)  #231216
      
      self.point_hl_pub = rospy.Publisher("/target_hl", Float64MultiArray, queue_size = 10)
      self.point_hlHSV_pub = rospy.Publisher("/target_hlHSV", Float64MultiArray, queue_size = 10)

      
      if self.usingUAV == "DJI":
        self.subscriber = rospy.Subscriber("/image/compressed", CompressedImage, self.image_cb,  queue_size = 1, buff_size=2**24)
        #self.subscriber = rospy.Subscriber("/dji/image", Image, self.image_cb, queue_size=1, buff_size=2**24)
        self.status_sub = rospy.Subscriber("/dji/status", String, self._status_cb, queue_size = 1) 
      elif self.usingUAV == "Bebop":
        self.subscriber = rospy.Subscriber("/bebop/image_raw", Image, self.image_cb, queue_size = 1, buff_size=2**24)
        self.sub_sensor_altitude = rospy.Subscriber("/bebop/states/ardrone3/PilotingState/AltitudeChanged", Ardrone3PilotingStateAltitudeChanged, self.altitudeCallback, queue_size=1)
        self.sub_sensor_gps = rospy.Subscriber("/bebop/states/ardrone3/PilotingState/PositionChanged", Ardrone3PilotingStatePositionChanged, self.gpsCallback, queue_size=1)
        self.sub_sensor_roll_yaw_pitch = rospy.Subscriber("/bebop/states/ardrone3/PilotingState/AttitudeChanged", Ardrone3PilotingStateAttitudeChanged, self.rollyawpitchCallback, queue_size=1)
      ###############################################################################
      
      # file write
      self.title  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + self.usingUAV)
      self.fourcc = cv2.VideoWriter_fourcc('X',"V",'I','D')
      self.out = cv2.VideoWriter(self.title + '.avi', self.fourcc, 20,(self.width, self.height))
      self.fourcc1 = cv2.VideoWriter_fourcc('X',"V",'I','D')
      self.out1 = cv2.VideoWriter(self.title + '_ori.avi', self.fourcc1, 20,(self.width, self.height))
        
      self.write_data_file = open(self.title+"_data.csv", 'w')
      self.write_data_file.write("time,lat,lot,alt,yaw,dis,dx,dy,inh,vz,vx,vy,deltax,deltay\n")
      self.count = 0
       
      self.img = np.array([])
      self.last_img = np.array([])
      self.last_frame = np.array([])
      self.t = 0
      self.FindTag = False
      self.target = [0,0]
      self.corner = []
      self.tag_count = 0
      self.last_point_x = [[]]*9
      self.last_point_y = [[]]*9
      self.last_alt = -1
      self.ini_db = {}
      self.old_center = [0,0]
      self.tag_id = -1
      self.last_target = [0,0]
      self.last_good_count = 0
      self.build_height = 0
      self.new_height = 0
      
      ###############################################################################
      
      # method
      if self.usingGrid == "cir":
        self.now_img = [None]*9
        self.old_img = [None]*9
        #self.template = np.load("template.npy")
        #self.area_in = [0]*4
        #self.area_out = [0]*4
        #self.out_prob = [0]*4
        #self.in_prob = [0]*4
        #self.max_area_in = 0
        #self.max_area_out = 0
        
        #self.contour_in = [None]*4
        #self.contour_out = [None]*4
        #self.contour_center = None
        #self.in_ignore_index = []
        #self.out_ignore_index = []

        #if self.usingVTS == True:
        #  self.direction_text = ["right", "forward", "left", "backward"]
        #  self.out_last_max_index = 1 # forward, 2 left, 3 backward
        #  self.in_last_max_index = 1  # forward 
        #  self.in_ignore_index = [3]    
        #elif self.usingVTS == False:
        #  self.direction_text = ["right", "forward", "left", "backward"]
        #  self.out_last_max_index = 0 # right
        #  self.in_last_max_index = 0 # right 
        #  self.in_ignore_index = [2] 
       
      elif self.usingGrid == "rec":
        self.now_img = [None]*9
        self.old_img = [None]*9
        #self.area_in = [0]*9 
        #self.in_prob = [0]*9
        #self.max_area_in = 0
        #self.contour_in = [None]*9
        #self.in_ignore_index = [4]
         
        #if self.usingVTS == True:
        #  self.direction_text = ["left forward", "forward", "right forward", "left", "center", "right","left backward", "backward", "right backward"]
        #  #self.out_last_max_index = 1 # forward, 2 left, 3 backward
        #  self.in_last_max_index = 1  # forward   
        #  self.in_ignore_index = [4,7,6,8] 
        #elif self.usingVTS == False:
        #  self.direction_text = ["left forward", "forward", "right forward", "left", "center", "right","left backward", "backward", "right backward"]
        #  #self.out_last_max_index = 5 # right
        #  self.in_last_max_index = 5 # right 
        #  self.in_ignore_index = [4,3,0,6]
      ###############################################################################
           
      # dji data
      if self.usingUAV == "DJI":
        self.isFlying = False
        self.isConnected = False
        self.areMotorsOn = False

        self.home = [500.0, 500.0]
        self.battery = 100

        self.yaw = 0.0
        self.lat = 500.0
        self.lot = 500.0
        self.alt = 0.001
        self.raw_alt = 0.001
        self.indoor_height = 0.001
        self.last_yaw = 0.0
        
        self.gimbal_pitch = 0.0  # up 29.2  ~  -90 down
        self.gimbal_yaw = 0.0
        self.gimbal_roll = 0.0
        self.intrinsic_parameters = np.array([661.705322, 0.000000, 403.230230, 0.000000, 511.792480, 236.514786, 0.000000, 0.000000, 1.000000]).reshape([3,3])
      # bebop data
      elif self.usingUAV == "Bebop":
        self.yaw = 0.0
        self.lat = 500.0
        self.lot = 500.0
        self.alt = 0.001 
        self.gimbal_pitch = -30
        self.intrinsic_parameters = np.array([537.292878, 0.000000, 427.331854, 0.000000, 527.000348, 240.226888, 0.000000, 0.000000, 1.000000]).reshape([3,3])
      self.nav_intrinsic_parameters = inv(self.intrinsic_parameters)
      self.gne = np.array([0,1,0,-1,0,0,0,0,self.alt]).reshape([3,3])
      self.nav_gne = inv(self.gne)
      self.last_gne = np.array([0,1,0,-1,0,0,0,0,self.alt]).reshape([3,3])
      self.last_nav_gne = inv(self.last_gne)
      ###############################################################################
      
    def _status_cb(self, msg):
      temp_data = msg.data.split(";")
      #for i in range(0, len(temp_data)):
      self.home = [float(temp_data[0].split("=")[1]), float(temp_data[1].split("=")[1])]  
      self.yaw = float(temp_data[3].split("=")[1])
      self.battery = float(temp_data[4].split("=")[1])
      self.isConnected = strtobool(temp_data[5].split("=")[1])
      self.areMotorsOn = strtobool(temp_data[6].split("=")[1])
      self.isFlying = strtobool(temp_data[7].split("=")[1])
      self.lat = float(temp_data[8].split("=")[1])
      self.lot = float(temp_data[9].split("=")[1])
      self.alt = float(temp_data[10].split("=")[1])
      self.raw_alt = float(temp_data[10].split("=")[1])
      if float(temp_data[14].split("=")[1]) != 0.0:
        self.alt = float(temp_data[14].split("=")[1])
      self.gimbal_pitch = float(temp_data[12].split("=")[1])
      self.gimbal_yaw = float(temp_data[13].split("=")[1])
      self.gimbal_roll = float(temp_data[11].split("=")[1])
      self.indoor_height = float(temp_data[14].split("=")[1])   
      self.velocityZ = float(temp_data[17].split("=")[1])
      self.velocityX = float(temp_data[18].split("=")[1])
      self.velocityY = float(temp_data[19].split("=")[1])

      rot_mat = np.array([math.cos(math.radians(self.gimbal_pitch+90)), -math.sin(math.radians(self.gimbal_pitch+90)), math.sin(math.radians(self.gimbal_pitch+90)), math.cos(math.radians(self.gimbal_pitch+90))])
      #gne1 = np.array([rot_mat[0],0,rot_mat[2],0,1,0,rot_mat[1],0,rot_mat[3]]).reshape([3,3])
      
      #231218更
      #----------------------------------------------------------
      if 0 < self.indoor_height <= 5:
        self.build_height = self.raw_alt - self.indoor_height
        self.new_height = self.raw_alt - self.build_height
      else:
        pass
      #----------------------------------------------------------
      
      #self.last_gne = self.gne
      self.last_nav_gne = self.nav_gne
      if self.indoor_height != 0.0:
        self.gne = np.array([rot_mat[0],rot_mat[1],0,rot_mat[2],rot_mat[3],0,0,0, (self.indoor_height)*(1/math.sin(math.radians(-self.gimbal_pitch)))]).reshape([3,3])
      else:
        self.gne = np.array([rot_mat[0],rot_mat[1],0,rot_mat[2],rot_mat[3],0,0,0, (self.new_height)*(1/math.sin(math.radians(-self.gimbal_pitch)))]).reshape([3,3])
      #self.gne = np.dot(gne1,gne2)
      self.nav_gne = inv(self.gne) 
    
    def start_detect_landmark_cb(self, msg):
      self.start_detect_landmark = msg.data
    
    #231216
    def start_take_image_cb(self, msg):
      self.start_take_image = msg.data
      
    def altitudeCallback(self, msg):
        self.alt = msg.altitude 
        rot_mat = np.array([math.cos(math.radians(self.gimbal_pitch+90)), -math.sin(math.radians(self.gimbal_pitch+90)), math.sin(math.radians(self.gimbal_pitch+90)), math.cos(math.radians(self.gimbal_pitch+90))])
      #gne1 = np.array([rot_mat[0],0,rot_mat[2],0,1,0,rot_mat[1],0,rot_mat[3]]).reshape([3,3])
      
        #self.last_gne = self.gne
        self.last_nav_gne = self.nav_gne
        
        self.gne = np.array([rot_mat[0],rot_mat[1],0,rot_mat[2],rot_mat[3],0,0,0, (self.alt)*(1/math.sin(math.radians(-self.gimbal_pitch)))]).reshape([3,3])
        #self.gne = np.dot(gne1,gne2)
        self.nav_gne = inv(self.gne) 
        
    def rollyawpitchCallback(self, msg):
        self.yaw =  msg.yaw 
    
    def gpsCallback(self, msg):
        self.lat = msg.latitude
        self.lot = msg.longitude
    
    def transelt_point_reverse(self, x, y):
      image_point = np.array([ x, y, 1 ])
      a = np.dot(self.intrinsic_parameters, self.gne)
      b = np.dot(a, image_point)
      b=b/b[2]
      print(b)
      return b[0],b[1] 
      
    def transelt_point(self,x,y):
      image_point = np.array([ x,y,1])
      a = np.dot(self.nav_gne,self.nav_intrinsic_parameters)
      b= np.dot(a,image_point)
      #print(b)
      b=b/b[2]
      return b[0],b[1]
    
    def transelt_point_new(self,x,y, nav_gne):
      image_point = np.array([ x,y,1])
      a = np.dot(nav_gne,self.nav_intrinsic_parameters)
      b= np.dot(a,image_point)
      #print(b)
      b=b/b[2]
      return b[0],b[1]
    
    def transelt_point_last(self, x, y):
      image_point = np.array([ x,y,1])
      a = np.dot(self.last_nav_gne,self.nav_intrinsic_parameters)
      b= np.dot(a,image_point)
      #print(b)
      b=b/b[2]
      return b[0],b[1]
    
    #把影像切小塊
    # def img_spilt(self, img, w, h):
    #   if self.usingGrid == "cir":
    #     s = [None]*9
    #     for i in range(1,len(s)+1):
    #       temp = np.zeros(img.shape[:2], dtype=np.uint8)
    #       mask = np.where(self.template == i)
    #       temp[mask] = 1
    #       s[i - 1] = cv2.bitwise_and(img,img, mask = temp)
    #       show = s[i - 1].copy()
    #       #show = cv2.resize(show, (86,48))
    #       #cv2.imshow("s"+str(i), show)
    #     #cv2.waitKey(0)  
    #     return s
    #   elif self.usingGrid == "rec":
    #     #pic = np.array(img * 255, dtype = np.uint8)

    #     split_img = [None]*9
    #     centerx = int(self.width/2)
    #     centery = int(self.height/2)
    #     #print(0,centery - int(h/2), centerx - int(w/2),centerx + int(w/2))
    #     split_img[0] = img[0:centery - int(h/2), 0:centerx - int(w/2)]
    #     split_img[1] = img[0:centery - int(h/2), centerx - int(w/2):centerx + int(w/2)]
    #     split_img[2] = img[0:centery - int(h/2), centerx + int(w/2):int(self.width)]
      
    #     split_img[3] = img[centery - int(h/2):centery + int(h/2), 0:centerx - int(w/2)]
    #     split_img[4] = img[centery - int(h/2):centery + int(h/2), centerx - int(w/2):centerx + int(w/2)]
    #     split_img[5] = img[centery - int(h/2):centery + int(h/2), centerx + int(w/2):int(self.width)]
      
    #     split_img[6] = img[centery + int(h/2):int(self.height), 0:centerx - int(w/2)]
    #     split_img[7] = img[centery + int(h/2):int(self.height), centerx - int(w/2):centerx + int(w/2)]
    #     split_img[8] = img[centery + int(h/2):int(self.height), centerx + int(w/2):int(self.width)]
    #     #cv2.imshow("mask", img)
    #     #for i in range(0, 9):
    #     #  cv2.imshow("s"+str(i), split_img[i])
    #     #cv2.waitKey(0)
    #     #for i in range(0,9):
    #     #  p = i % 3
    #     #  level = int(i / 3)
    #     #  split_img[i] = pic[int(h/3)*int(level):int(h/3)*int(level+1), int(w/3)*int(p):int(w/3)*int(p+1)]
    #     return split_img

    # def img_offset_compute(self, max_index, w, h):
    #   split_index = [0]*9
    #   centerx = int(self.width/2)
    #   centery = int(self.height/2)
    #   split_index[0] = (0                 ,0)
    #   split_index[1] = (centerx - int(w/2),0)
    #   split_index[2] = (centerx + int(w/2),0)
      
    #   split_index[3] = (0                 ,centery - int(h/2))
    #   split_index[4] = (centerx - int(w/2),centery - int(h/2))
    #   split_index[5] = (centerx + int(w/2),centery - int(h/2))
      
    #   split_index[6] = (0                 ,centery + int(h/2))
    #   split_index[7] = (centerx - int(w/2),centery + int(h/2))
    #   split_index[8] = (centerx + int(w/2),centery + int(h/2))
    #   """x_offset = 0
    #   y_offset = 0
    #   p = max_index % 3
    #   level = int(max_index / 3)"""
    #   return split_index[max_index]    
      
    # def reject_outliers(self, data, m=2):
    #   return data[abs(data - np.mean(data)) < m * np.std(data)]
    
    # def re_index(self, index):
    #   return 8 - index
    
    # def rotate_point(self, x, y, centerx, centery):
    #   angle = -(self.gimbal_yaw-self.last_yaw)*math.pi/180
    #   r_fx = ((x - centerx) * math.cos(angle)) - ((y - centery) * math.sin(angle))
    #   r_fy = ((x - centerx) * math.sin(angle)) + ((y - centery) * math.cos(angle))
    #   return r_fx + centerx, r_fy + centery
    
    # def union_rec(self, a, b):
    #   x = min(a[0], b[0])
    #   y = min(a[1], b[1])
    #   w = max(a[0]+a[2], b[0]+b[2]) - x
    #   h = max(a[1]+a[3], b[1]+b[3]) - y
    #   return (x - int(self.width/10), y-int(self.height/10), w+int(self.width/5), h+int(self.height/5))
    
    # def get_image_center(self):
    #   now_h = self.alt
      
    #   if now_h < 5:
    #     return self.height / 4.0 * 3
    #   elif now_h > 70:
    #     return self.height / 2.0
    #   else:
    #     # 5a + b = self.height / 4.0 * 3
    #     # 70a + b = self.height / 2.0
    #     # 65a = self.height / 2.0 - self.height / 4.0 * 3
    #     # a = (self.height / 2.0 - self.height / 4.0 * 3) / 65.0
    #     # b = self.height / 2.0 - 70*a
    #     a = (self.height / 2.0 - (self.height / 4.0 * 3.0)) / 65.0
    #     b = self.height / 2.0 - 70.0*a
    #     return a*now_h + b

    # def img2hist_scan(self, img):
    #     h, w, c = img.shape
    #     bin_size = 64
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     hist = cv2.calcHist([img],[0, 1], None, [bin_size, bin_size], [0, 256, 0, 256] )
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
    #     cv2.filter2D(hist,-1,kernel,hist)
    #     cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    #     hist = cv2.calcHist([img],[0, 1], self.get_mask(img, hist ), [bin_size, bin_size], [0, 256, 0, 256] )
    #     cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    #     return hist

    # def get_mask(self, img, hist):
    #     #bin_size = 128
    #     dst = cv2.calcBackProject([img],[0,1],hist,[0,256,0,256],1)
    #     disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # smoothing
    #     cv2.filter2D(dst,-1,disc,dst) # smoothing
    #     ret, mask = cv2.threshold(dst, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     #mask = cv2.merge((mask,mask,mask))
    #     return mask
        
    # def do_fm(self, img, isFirst = False):
    #   img_viz = img.copy()
      
    #   self.FirstTrack = True
    #   #split_img = self.img_spilt(img, int(self.width/3), int(self.height/3))
    #   #old_split_img = self.img_spilt(self.last_img, int(self.width/3), int(self.height/3))
    #   #isZero = False
    #   #now_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #   #now_blurred = cv2.GaussianBlur(now_gray, (11, 11), 0) 
    #   #now_binaryIMG = cv2.Canny(now_blurred, 20, 160)
    #   #(cnts, _) = cv2.findContours(now_binaryIMG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    #   #max_contour = max(cnts, key = cv2.contourArea)
    #   #x,y,w,h = cv2.boundingRect(max_contour)
    #   #center = (x+int(w/2), y+int(h/2))
    #   #cv2.rectangle(img_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #   if isFirst == True:
        
    #     # video 1
    #     left = (int(self.corner[1][0]), int(self.corner[1][1]))
    #     right_up = (int(self.corner[0][0]), int(self.corner[0][1]))
    #     left_down = (int(self.corner[2][0]), int(self.corner[2][1]))
    #     right_down = (int(self.corner[3][0]), int(self.corner[3][1]))
        
    #     #cv2.line(self.last_img, left, left_down, (0,0,255), 3)
    #     #cv2.line(self.last_img, right_down, left_down, (0,255,0), 3)
    #     #cv2.line(self.last_img, right_down, right_up, (255,0,0), 3)
        
    #     # 避免順序錯亂
    #     min_left = min(left[0], left_down[0], right_up[0], right_down[0])
    #     max_right = max(left[0], left_down[0], right_up[0], right_down[0])
    #     min_top = min(left[1], right_up[1], left_down[1], right_down[1])
    #     max_down = max(left[1], right_up[1], left_down[1], right_down[1])
        
    #     print("111",min_left, max_right, min_top, max_down)
    #     #cv2.rectangle(self.last_img, (min_left, min_top), (max_right, max_down), (0, 255, 255), 2)
    #     #cv2.imshow("last", self.last_img)
    #     #cv2.imshow("img", img)
    #     #cv2.waitKey(0)
    #     left_w = int(max_right - min_left)
    #     left_h = int(max_down - min_top)
    #     #left_w = int(math.sqrt((self.corner[2][0] - self.corner[1][0])**2 + (self.corner[2][1] - self.corner[1][1])**2))
    #     #left_h = int(math.sqrt((self.corner[2][0] - self.corner[3][0])**2 + (self.corner[2][1] - self.corner[3][1])**2))
    #     left_center = (self.target[0], self.target[1])
    #     # video 2
    #     #left = (int(self.corner[3][0]), int(self.corner[3][1]))
    #     #left_w = int(math.sqrt((self.corner[3][0] - self.corner[0][0])**2 + (self.corner[3][1] - self.corner[0][1])**2))
    #     #left_h = int(math.sqrt((self.corner[3][0] - self.corner[2][0])**2 + (self.corner[3][1] - self.corner[2][1])**2))
    #     #left_center = (self.target[0], self.target[1])        
        
    #     self.initBB = (min_left, min_top, left_w, left_h)
    #     print("1,", self.initBB)
    #     self.tracker = cv2.TrackerCSRT_create() 
    #     self.tracker.init(self.last_img, self.initBB)
        
    #     (success, box) = self.tracker.update(img)
    #     (x, y, w, h) = [int(v) for v in box]
    #     print((x, y, w, h))
    #     #success = False
        
    #     if x <= 0 and y <= 0:
    #       print("First Time Track Error...")
    #     else:
    #       last_hist = self.img2hist_scan(self.last_img[min_top:min_top+left_h, min_left:min_left+left_w])
    #       now_hist = self.img2hist_scan(img[y:y+h, x:x+w])
    #       another_hist = self.img2hist_scan(img[:h,:w])

    #       score = cv2.compareHist(last_hist, now_hist, cv2.HISTCMP_CORREL)
    #       ascore = cv2.compareHist(last_hist, another_hist, cv2.HISTCMP_CORREL)
          
    #       cv2.imshow("last_hist", last_hist)
    #       cv2.imshow("now_hist", now_hist)
    #       cv2.imshow("another_hist", another_hist)
    #       #gray_last_roi = cv2.cvtColor(self.last_img[left[1]:left[1]+left_h, left[0]:left[0]+left_w], cv2.COLOR_BGR2GRAY)
    #       #gray_now_roi = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    #       #(score, diff) = compare_ssim(gray_last_roi, gray_now_roi, full=True)
    #       print("True", score, ascore)
        
    #       if score > 0.7:
    #         cv2.rectangle(img_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         self.target =  [x + w*0.5, y+h*0.5, x, y, x, y+h, x+w, y+h, x+w, y] 
    #         self.last_img = img
    #         self.last_alt = self.alt
    #         self.last_yaw = self.gimbal_yaw
    #         print("Tracker Success")
    #       else: #FM
    #         last_gray = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2GRAY)
    #         now_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         # find the keypoints and descriptors with SIFT
    #         kp1, des1 = self.orb.detectAndCompute(last_gray,None)
    #         kp2, des2 = self.orb.detectAndCompute(now_gray,None)
          
    #         #old = []
    #         #now = []
    #         good = []
    #         if des1 is not None and des2 is not None: 
    #           matches = self.bf.knnMatch(des1,des2, k=2)

    #           for m in matches:
    #             if len(m) == 2:
    #               if m[0].distance < 0.75*m[1].distance:
    #                 #good.append([m])
            
    #                 img1_idx = m[0].queryIdx
    #                 img2_idx = m[0].trainIdx
    #                 (x1, y1) = kp1[img1_idx].pt
    #                 (x2, y2) = kp2[img2_idx].pt
    #                 #offset = self.img_offset_compute(i, int
    #                 if abs(x2 - x1) < (self.width/10) and abs(y2 - y1) < self.height/10:
    #                   cv2.circle(img_viz, (int(x1),int(y1)), 5, (255,0,0), -1)
    #                   cv2.circle(img_viz, (int(x2),int(y2)), 4, (0,255,0), -1)

    #                   good.append(m[0])
    #                   #old.append([x1,y1])
    #                   #now.append([x2,y2])
                    
    #                   #if len(good) == 10:
    #                   #  break
    #         print("good",len(good))
    #         if len(good) > 0:
    #           pts_old = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #           pts_now = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #           M, mask = cv2.findHomography(pts_old, pts_now, cv2.RANSAC,5.0)
    #           if M is None: # 0609
    #               return   
    #           #print("====================================")
    #           #print(M)
    #           last_target = np.float32([ [left_center[0], left_center[1]], [min_left,min_top],[min_left,max_down],[max_right,max_down],[max_right,min_top] ]).reshape(-1,1,2)
    #           #print(last_target)
    #           dst = cv2.perspectiveTransform(last_target,M)
    #           #print(dst)
    #           #print("====================================")
    #           guess_center = (dst[0][0][0], dst[0][0][1], self.width/2, self.height/4*3)
    #           guess_left_up = (dst[1][0][0], dst[1][0][1], guess_center[0], guess_center[1])
    #           guess_left_down = (dst[2][0][0], dst[2][0][1], guess_center[0], guess_center[1])
    #           guess_right_down = (dst[3][0][0], dst[3][0][1], guess_center[0], guess_center[1])
    #           guess_right_up = (dst[4][0][0], dst[4][0][1], guess_center[0], guess_center[1])
    #           # = ( (guess_left_up[0]+guess_left_down[0]+guess_right_up[0]+guess_right_down[0])/4.0, (guess_left_up[1]+guess_left_down[1]+guess_right_up[1]+guess_right_down[1])/4.0)          
    #           #cv2.rectangle(img_viz, (int(guess_x), int(guess_y)), (int(guess_x + guess_w), int(guess_y + guess_h)), (0, 255, 0), 2)
    #           cv2.polylines(img_viz,[np.int32(dst)], True, (0,255,0), 2, cv2.LINE_AA)
            
    #           self.target =  [guess_center[0], guess_center[1], guess_left_up[0], guess_left_up[1], guess_left_down[0], guess_left_down[1], guess_right_down[0], guess_right_down[1], guess_right_up[0], guess_right_up[1]] 
    #           #self.union_data = (x, y, w, h)
    #           #self.guess_data = (guess_x, guess_y, guess_w, guess_h)
    #           self.last_img = img  
    #           self.last_alt = self.alt            
    #           self.last_yaw = self.gimbal_yaw
              
    #           left_w = int(math.sqrt((self.target[2] - self.target[8])**2 + (self.target[3] - self.target[9])**2))
    #           left_h = int(math.sqrt((self.target[2] - self.target[4])**2 + (self.target[3] - self.target[5])**2))
              
    #           self.initBB = (int(self.target[2]), int(self.target[3]), left_w, left_h)
    #           self.tracker = cv2.TrackerCSRT_create() 
    #           self.tracker.init(img, self.initBB)
    #         else:
    #           print("FM no target...")
    #           #self.last_img = img         
                    
    #   elif isFirst == False:
    #     (success, box) = self.tracker.update(img)
    #     #success = False
    #     #if self.alt > 50:
    #     #  success = False
    #     (x, y, w, h) = [int(v) for v in box]
    #     #print(success,(x, y, w, h))
    #     print((x, y, w, h))
    #     #success = False
    #     left_w = int(math.sqrt((self.target[2] - self.target[8])**2 + (self.target[3] - self.target[9])**2))
    #     left_h = int(math.sqrt((self.target[2] - self.target[4])**2 + (self.target[3] - self.target[5])**2))
    #     #left_h = self.target
    #     #gray_last_roi = cv2.cvtColor(self.last_img[self.target[3]:self.target[3]+left_h, self.target[2]:self.target[2]+left_w], cv2.COLOR_BGR2GRAY)
    #     #gray_now_roi = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    #     #(score, diff) = compare_ssim(gray_last_roi, gray_now_roi, full=True)
    #     score = 0
    #     ascore = 0
    #     if x <= 0 and y <= 0:
    #       print("tracker error")
    #     else:
    #       last_hist = self.img2hist_scan(self.last_img[int(self.target[3]):int(self.target[3]+left_h), int(self.target[2]):int(self.target[2]+left_w)])
    #       now_hist = self.img2hist_scan(img[y:y+h, x:x+w])
    #       score = cv2.compareHist(last_hist, now_hist, cv2.HISTCMP_CORREL)
    #       another_hist = self.img2hist_scan(img[:h,:w])
    #       ascore = cv2.compareHist(last_hist, another_hist, cv2.HISTCMP_CORREL)

    #       cv2.imshow("last_hist", last_hist)
    #       cv2.imshow("now_hist", now_hist)
    #       cv2.imshow("another_hist", another_hist)
    #     print("False", score, ascore)
        
    #     if score > 0.7:
    #       (x, y, w, h) = [int(v) for v in box]
    #       cv2.rectangle(img_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #       self.target =  [x + w*0.5, y+h*0.5, x, y, x, y+h, x+w, y+h, x+w, y] 
    #       self.last_target = self.target
    #       self.last_img = img
    #       self.last_alt = self.alt 
    #       self.last_yaw = self.gimbal_yaw
    #       print("Tracker Success")
    #     else: #FM
    #       cv2.imshow("last_hist", last_hist)
    #       cv2.imshow("now_hist", now_hist)
    #       last_gray = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2GRAY)
    #       now_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #       # find the keypoints and descriptors with SIFT
    #       kp1, des1 = self.orb.detectAndCompute(last_gray,None)
    #       kp2, des2 = self.orb.detectAndCompute(now_gray,None)
          
    #       #old = []
    #       #now = []
    #       good = []
    #       if des1 is not None and des2 is not None: 
    #         matches = self.bf.knnMatch(des1,des2, k=2)

    #         for m in matches:
    #           if len(m) == 2:
    #             if m[0].distance < 0.75*m[1].distance:
    #               #good.append([m])
            
    #               img1_idx = m[0].queryIdx
    #               img2_idx = m[0].trainIdx
    #               (x1, y1) = kp1[img1_idx].pt
    #               (x2, y2) = kp2[img2_idx].pt
    #               #offset = self.img_offset_compute(i, int
    #               #if abs(x2 - x1) < (self.width/10) and abs(y2 - y1) < self.height/10:
    #               cv2.circle(img_viz, (int(x1),int(y1)), 5, (255,0,0), -1)
    #               cv2.circle(img_viz, (int(x2),int(y2)), 4, (0,255,0), -1)

    #               good.append(m[0])
    #               #old.append([x1,y1])
    #               #now.append([x2,y2])
                    
    #               #if len(good) == 10:
    #               #  break
    #       print("good",len(good)) # 匹配吻合特徵點的數量
    #       if len(good) > 0:
    #         pts_old = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #         pts_now = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #         M, mask = cv2.findHomography(pts_old, pts_now, cv2.RANSAC,5.0)   
    #         if M is None:
    #             return
    #         #print("=========================")
    #         #print(M)
    #         last_target = np.float32([ [self.target[0],self.target[1]],[self.target[2],self.target[3]],[self.target[4],self.target[5]],[self.target[6],self.target[7]],[self.target[8],self.target[9]] ]).reshape(-1,1,2)
    #         dst = cv2.perspectiveTransform(last_target,M)
    #         #print(last_target)
    #         #print(dst)
    #         #print("=========================")
    #         guess_center = (dst[0][0][0], dst[0][0][1], self.width/2, self.height/4*3)
    #         guess_left_up = (dst[1][0][0], dst[1][0][1], guess_center[0], guess_center[1])
    #         guess_left_down = (dst[2][0][0], dst[2][0][1], guess_center[0], guess_center[1])
    #         guess_right_down = (dst[3][0][0], dst[3][0][1], guess_center[0], guess_center[1])
    #         guess_right_up = (dst[4][0][0], dst[4][0][1], guess_center[0], guess_center[1])
    #         # = ( (guess_left_up[0]+guess_left_down[0]+guess_right_up[0]+guess_right_down[0])/4.0, (guess_left_up[1]+guess_left_down[1]+guess_right_up[1]+guess_right_down[1])/4.0)       
    #         #cv2.rectangle(img_viz, (int(guess_x), int(guess_y)), (int(guess_x + guess_w), int(guess_y + guess_h)), (0, 255, 0), 2)
    #         cv2.polylines(img_viz,[np.int32(dst)], True, (0,255,0), 2, cv2.LINE_AA)
            
    #         self.target =  [guess_center[0], guess_center[1], guess_left_up[0], guess_left_up[1], guess_left_down[0], guess_left_down[1], guess_right_down[0], guess_right_down[1], guess_right_up[0], guess_right_up[1]]
    #         #self.union_data = (x, y, w, h)
    #         #self.guess_data = (guess_x, guess_y, guess_w, guess_h)
    #         #cv2.imshow("last", self.last_img)
    #         #self.last_img = img
    #         self.last_img = img 
    #         #cv2.imshow("img", self.last_img) 
    #         #cv2.waitKey(0)
    #         self.last_alt = self.alt            
    #         self.last_yaw = self.gimbal_yaw       
    #         #self.last_alt = self.alt 
            
    #         left_w = int(math.sqrt((self.target[2] - self.target[8])**2 + (self.target[3] - self.target[9])**2))
    #         left_h = int(math.sqrt((self.target[2] - self.target[4])**2 + (self.target[3] - self.target[5])**2)) 
    #         """if left_w != 1 and left_h != 1:                  
    #           self.initBB = (int(self.target[2]), int(self.target[3]), left_w, left_h)
    #           print(self.initBB)
    #           self.tracker = cv2.TrackerCSRT_create() 
    #           #print(self.initBB)
    #           #print(self.img)
    #           self.tracker.init(img, self.initBB)
    #         else:
    #           self.initBB = (int(self.target[2])-7, int(self.target[3])-7, left_w+7, left_h+7)
    #           print(self.initBB)
    #           self.tracker = cv2.TrackerCSRT_create() 
    #           #print(self.initBB)
    #           #print(self.img)
    #           self.tracker.init(img, self.initBB)"""
    #         #cv2.imshow("now", img)
    #         #cv2.imshow("now1", img_viz)
    #         #cv2.waitKey(0)
    #       else:
    #         print("FM no target...")
          
    #   #t = rospy.get_time()
    #   #if t % 2 == 0:   
    #   #  self.old_img = self.now_img
    #   #  self.last_alt = self.alt
    #   #cv2.imshow("img_viz", img_viz)
    #   #cv2.waitKey(0)
    #   return img_viz
    
    #圖像、視覺化圖像
    # def hsv(self, img, img_viz, x ,y, w, h):

    #     #start = time.time()    # 計時開始
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # 將BGR轉換為HSV格式
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # erosion kernel        
        
    #     # 定義白色HSV的範圍
    #     lower_white = np.array([0, 0, 237])
    #     upper_white = np.array([180, 30, 255])
        
    #     mask = cv2.inRange(hsv, lower_white, upper_white)    # 進行HSV過濾
    #     res = cv2.bitwise_and(img, img, mask=mask)    # 將過濾後的二值圖像與原始圖像進行位元運算
    #     gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)    # 將過濾後的圖像轉換為灰階圖像
    #     gray = cv2.erode(gray, kernel)    # 侵蝕，將白色小圓點移除
    #     contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # 查找輪廓
        
    #     # 找出目標輪廓
    #     target_area = 0
    #     order = 0
    #     find = False
    #     for i in range(len(contours)):
    #         # 計算面積
    #         area = cv2.contourArea(contours[i])
            
    #         if self.hsv_detect:
    #             if area > 4300:
    #                 self.apt_detect = True # 目標切換成Apriltag
    #             elif area > 300:
    #                 rect = cv2.minAreaRect(contours[i])
    #                 if rect[1][1] / rect[1][0] < 14/10 and rect[1][1] / rect[1][0] > 10/14: # 長寬比篩選
    #                     if area > self.HSVSize * 0.9 and area < self.HSVSize * 1.25: # 0531 面積紀錄篩選
    #                         find = True
    #                         self.HSVSize, target_area = area, area
    #                         order = i  
    #                #     else:
    #                #         print("OldSize: ", str(self.HSVSize), "NowSize: ", str(area))
    #                #         print("Size Bug")
    #                # else:
    #                #     print("長寬： ", rect[1][1], rect[1][0])
                            
    #         else: # 確認hsv與模板匹配結果相符
    #             if area > 300:
    #                 M = cv2.moments(contours[i])
    #                 cx = int(M['m10'] / M['m00'])
    #                 cy = int(M['m01'] / M['m00'])
    #                 if x != -1 and y != -1:
    #                     if x <= cx <= x+w and y <= cy <= y+h:
    #                         #print("hsv + pm OK!")
    #                         find = True
    #                         self.hsv_detect = True
    #                         target_area = area
    #                         order = i 
                              
    #                         # 0608  
    #                         self.HSVSize = area   
    #                         print("FirstSize: ", area)
    #                         continue
                 
        
    #     if target_area != 0:        
    #         # 找出輪廓並繪製
    #         rect = cv2.minAreaRect(contours[order])
    #         box = cv2.boxPoints(rect)
    #         box = box.astype(int)
    #         #box = np.array([])
    #         box = np.array([box[2], box[1], box[0], box[3]])
    #         print("box", box)
    #         cv2.drawContours(img_viz, [box], 0, (0, 255, 0), 2)
        
    #         # 找出中心點並繪製
    #         M = cv2.moments(contours[order])
    #         cx = int(M['m10'] / M['m00'])
    #         cy = int(M['m01'] / M['m00'])
    #         cv2.circle(img_viz, (cx, cy), 1, (0, 0, 255), 5)
    #         #print(target_area)
    #         return [(cx, cy), box]
    #     else:
    #         #print("Empty Detect")
    #         return []
    
    # def pm(self, template, image, img_viz):
    
    #     u = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # # 獲取模板圖像的寬度和高度
    #     h, w = t.shape
    
    # # 使用模板匹配算法進行搜索
    #     res = cv2.matchTemplate(u, t, cv2.TM_CCOEFF_NORMED)
    
    # # 設定閾值
    #     threshold = 0.75
    
    # # 搜尋位置
    #     locations = np.where(res >= threshold)
    #     locations = list(zip(*locations[::-1]))
    
    # # 顯示圖像
    #     if len(locations) > 0: 
    #         loc = locations[0]
    #         cv2.rectangle(img_viz, loc, (loc[0] + w, loc[1] + h), (0, 255, 0), 1)
    #         return img_viz, loc[0], loc[1], loc[0] + w, loc[1] + h
    #     else:
    #         return img_viz, -1, -1, -1, -1


    #計算圖像中心點至邊的最短距離(直接帶公式)
    def point_distance_line(self, point, line_point1, line_point2):
        #檢查直線兩點是否重疊，如果重疊則回傳到其中一點的距離
        if np.array_equal(line_point1, line_point2):
            point_array = np.array(point)
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array - point1_array)

        #Ax + By + C = 0
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]

        #點到直線垂直距離公式
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))

        # 計算座標點H
        H = ( B**2 * point[0] - A*B*point[1] - A*C ) / (A**2 + B**2)
        K = ( -A*B*point[0] + A**2 * point[1] - B*C ) / (A**2 + B**2)
        h_point = np.array([H, K])
        return distance, h_point

    def do_alg(self, img, output_path = "output_monodepth"):
        stime = time.time()
        # run DPT
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        image_input = transform({"image": image})["image"]
        
        with torch.no_grad():
            sample = torch.from_numpy(image_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
        )
        
        # Save the processed image
        output_image = (prediction * 255).clip(0, 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(output_path, output_image)
         
        if output_image is not None:
    
            font = cv2.FONT_HERSHEY_SIMPLEX
            resize_image = cv2.resize(output_image, (self.width, self.height))
            img_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
            
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
                # print("Approximated Polygon:","\n" ,approx_polygon)
                rect = cv2.minAreaRect(approx_polygon)

                #box用途:方便繪圖
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                #(x,y):外接矩形中心點 ; (w,h):寬、高 ; angle:旋轉角度
                (x, y), (w, h), angle = rect

                #計算外接矩形的中心座標
                center_x = int(x)
                center_y = int(y)
                print(' 外接矩形的中心座標點:', '(', center_x, ',', center_y, ')')

                #用途:方便計算座標點轉向位置
                half_width = int(w / 2)
                half_height = int(h / 2)

                #計算矩形的四個角的相對坐標
                corner1 = (-half_width, -half_height)  # 左上角
                corner2 = (half_width, -half_height)   # 右上角
                corner3 = (half_width, half_height)    # 右下角
                corner4 = (-half_width, half_height)   # 左下角

                #角度>0 : 向左
                #角度<0 : 向右
                print(' 旋轉角度 : ',round(float(angle), 3),'度')

                #將相對坐標旋轉回原始坐標
                cos_theta = np.cos(np.radians(angle))
                sin_theta = np.sin(np.radians(angle))

                #由於angle預設為正(往左旋轉直到回正)，
                #若angle > 45度下去做cos、sin的相對座標旋轉回原始座標
                #會造成原始座標判斷錯誤，故以此判斷式校正錯誤的座標
                if abs(angle) > 45:
                    rotated_corner2 = (
                        int(center_x + corner1[0] * cos_theta - corner1[1] * sin_theta),
                        int(center_y + corner1[0] * sin_theta + corner1[1] * cos_theta)
                    )
                    rotated_corner3 = (
                        int(center_x + corner2[0] * cos_theta - corner2[1] * sin_theta),
                        int(center_y + corner2[0] * sin_theta + corner2[1] * cos_theta)
                    )
                    rotated_corner4 = (
                        int(center_x + corner3[0] * cos_theta - corner3[1] * sin_theta),
                        int(center_y + corner3[0] * sin_theta + corner3[1] * cos_theta)
                    )
                    rotated_corner1 = (
                        int(center_x + corner4[0] * cos_theta - corner4[1] * sin_theta),
                        int(center_y + corner4[0] * sin_theta + corner4[1] * cos_theta)
                    )
                    print('∵angle > 45 , ∴進行校正')
                else:
                    # center_x、center_y : 讓旋轉前後的中心點一致
                    # [new_x] = R * [x] = [cos(Θ) -sin(Θ)] * [x]
                    # [new_y]       [y]   [sin(Θ)  cos(Θ)]   
                    rotated_corner1 = (
                        int(center_x + corner1[0] * cos_theta - corner1[1] * sin_theta),
                        int(center_y + corner1[0] * sin_theta + corner1[1] * cos_theta)
                    )
                    rotated_corner2 = (
                        int(center_x + corner2[0] * cos_theta - corner2[1] * sin_theta),
                        int(center_y + corner2[0] * sin_theta + corner2[1] * cos_theta)
                    )
                    rotated_corner3 = (
                        int(center_x + corner3[0] * cos_theta - corner3[1] * sin_theta),
                        int(center_y + corner3[0] * sin_theta + corner3[1] * cos_theta)
                    )
                    rotated_corner4 = (
                        int(center_x + corner4[0] * cos_theta - corner4[1] * sin_theta),
                        int(center_y + corner4[0] * sin_theta + corner4[1] * cos_theta)
                    )
                print('---------------------------------------------')

                # 轉換為整數座標
                rotated_corner1 = tuple(map(int, rotated_corner1))
                rotated_corner2 = tuple(map(int, rotated_corner2))
                rotated_corner3 = tuple(map(int, rotated_corner3))
                rotated_corner4 = tuple(map(int, rotated_corner4))

                #矩形四個角落的座標點
                print(' 矩形四個角落的座標點')
                print(' Rotated Corner 1:', rotated_corner1)
                print(' Rotated Corner 2:', rotated_corner2)
                print(' Rotated Corner 3:', rotated_corner3)
                print(' Rotated Corner 4:', rotated_corner4)
                print('---------------------------------------------')
                #-----------------------------------------------------------------------

                #計算中心點到四邊的最短距離、座標點
                top_distance, top_h_point = self.point_distance_line(image_center_point, rotated_corner1, rotated_corner2)
                bottom_distance, bottom_h_point = self.point_distance_line(image_center_point, rotated_corner3, rotated_corner4)
                left_distance, left_h_point = self.point_distance_line(image_center_point, rotated_corner1, rotated_corner4)
                right_distance, right_h_point = self.point_distance_line(image_center_point, rotated_corner2, rotated_corner3)

                print(' Distance to Top:', int(top_distance), '\n','Top垂點座標:', '(', int(top_h_point[0]),',', int(top_h_point[1]), ')', '\n')
                print(' Distance to Bottom:', int(bottom_distance), '\n','Bottom垂點座標:', '(', int(bottom_h_point[0]),',', int(bottom_h_point[1]), ')', '\n')
                print(' Distance to Left:', int(left_distance), '\n','Left垂點座標:', '(', int(left_h_point[0]),',', int(left_h_point[1]), ')', '\n')
                print(' Distance to Right:', int(right_distance), '\n','Right垂點座標:', '(', int(right_h_point[0]), ',', int(right_h_point[1]), ')', '\n')
                print('---------------------------------------------')
                min_distance, min_distance_direction = min(
                    (top_distance, 'Top'),
                    (bottom_distance, 'Bottom'),
                    (left_distance, 'Left'),
                    (right_distance, 'Right'),
                    key=lambda x: x[0]
                )

                if min_distance_direction == 'Top':
                    min_h_point = top_h_point
                elif min_distance_direction == 'Bottom':
                    min_h_point = bottom_h_point
                elif min_distance_direction == 'Left':
                    min_h_point = left_h_point
                else:
                    min_h_point = right_h_point

                print(' Minimum Distance:', min_distance)
                print(' Move to:', '『', min_distance_direction , '』', 'side')
                print(' Move to:', min_h_point)
                print('---------------------------------------------')

                #以下繪圖-------------------------------------------------------------------------------------------------------
        
                #影像中心點與各邊的垂線
                cv2.line(resize_image, image_center_point, tuple(top_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(resize_image, image_center_point, tuple(bottom_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(resize_image, image_center_point, tuple(left_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(resize_image, image_center_point, tuple(right_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)        
                #影像中心點與最短邊的垂線
                cv2.line(resize_image, image_center_point, tuple(min_h_point.astype(int)), (0, 150, 255), 12, cv2.LINE_AA)

                #最大輪廓
                cv2.drawContours(resize_image, [largest_contour], -1, (0, 255, 0), 4)   
                #最大多邊形
                cv2.drawContours(resize_image, [approx_polygon], -1, (0, 255, 255), 4)  
                #最大外接矩形
                cv2.drawContours(resize_image, [box], 0, (255, 0, 0), 5)                

                #外接矩形corner座標點
                # cv2.circle(resize_image, rotated_corner1, 3, (0, 0, 255), cv2.FILLED)  # Red
                # cv2.circle(resize_image, rotated_corner2, 3, (0, 255, 0), cv2.FILLED)  # Green
                # cv2.circle(resize_image, rotated_corner3, 3, (255, 0, 0), cv2.FILLED)  # Blue
                # cv2.circle(resize_image, rotated_corner4, 3, (0, 255, 255), cv2.FILLED)  # Yellow

                #影像中心點與各邊的垂足點
                cv2.circle(resize_image, tuple(top_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
                cv2.circle(resize_image, tuple(bottom_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
                cv2.circle(resize_image, tuple(left_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
                cv2.circle(resize_image, tuple(right_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
                #最短邊的垂足點
                cv2.circle(resize_image, tuple(min_h_point.astype(int)), 10, (0, 0, 255), cv2.FILLED)

                #影像中心點
                cv2.circle(resize_image, image_center_point, 15, (0, 0, 255), cv2.FILLED)

                #文字說明

                # cv2.putText(resize_image, 'Red:Center', (10, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # cv2.putText(resize_image, 'Green:Max Contour', (10, 50), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(resize_image, 'Blue:Max Rectangle', (10, 75), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(resize_image, 'Black:Foot Drop', (10, 100), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(resize_image, 'Orange:Perpendicular line', (10, 125), font, 0.5, (0, 110, 255), 1, cv2.LINE_AA)
                #於圖像顯示判斷結果
                # solution = 'Move to:'+ min_distance_direction + ' side'
                # cv2.putText(resize_image, solution, (150, 390), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                
                #231216
                self.target = min_h_point #告知無人機要移動到的座標點
    
                #
                d00x,d00y = self.transelt_point(int(self.width/2), int(self.height/2))
                d11x,d11y = self.transelt_point(int(self.width/2), int(self.height/2)+5)
                dis_ini = math.sqrt((d00x - d11x)**2 + (d00y - d11y)**2) * 100.0
                pixel_corr = 5 / dis_ini * 8.0

                now_image_height = self.height/2 + pixel_corr
                dy = int(self.target[1]) - int(now_image_height)
                #elif self.alt <= 20:
                #dy = int(self.target[1]) - int(self.height/4*3)

                d1x,d1y = self.transelt_point(self.target[0], self.target[1])

                #if self.alt > 20:
                d2x,d2y = self.transelt_point(int(self.width/2), int(now_image_height))
                #elif self.alt <= 20:
                #d2x,d2y = self.transelt_point(int(self.width/2), int(self.height/4*3))

                d3x,d3y = self.transelt_point(int(self.width/2), self.target[1])

                #if self.alt > 20:
                d4x,d4y = self.transelt_point(self.target[0], int(now_image_height))
                #elif self.alt <= 20:
                #  d4x,d4y = self.transelt_point(self.target[0], int(self.height/4*3))

                dis = math.sqrt((d2x - d1x)**2 + (d2y - d1y)**2) 
                dis1 = math.sqrt((d2x - d3x)**2 + (d2y - d3y)**2) 
                dis2 = math.sqrt((d2x - d4x)**2 + (d2y - d4y)**2) 
                cv2.putText(resize_image, "fps: " + str(int(1/(time.time()-stime))) + " frames", (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("img_viz", resize_image)
                self.out.write(resize_image)
                self.write_data_file.write(str(rospy.get_time())   + "," + str(self.lat) + ","       + str(self.lot) + "," + str(self.alt) + "," +
                                           str(self.gimbal_yaw)    + "," + str(dis) + ","            + str(dis2) + ","     + str(dis1) + "," +
                                           str(self.indoor_height) +  "\n") 
                                          #  str(self.velocityZ) + "," + str(self.velocityX) + "," +str(self.velocityY)  + "," + str(dx)   + "," + str(dy) + "\n") 
                self.point_hl_pub.publish(Float64MultiArray(data = [self.target[0], self.target[1], dis, dis2, dis1, now_image_height, self.new_height ]))
                cv2.waitKey(1)
            
            # if show_image_flag == True:
            #   stime = time.time()
            #   img_viz = img.copy()

            #   if self.apt_detect is False :
            #       if self.Target_Stage != "hsv":
            #           self.Target_Stage = "hsv"
            #           print("hsv")

            #       if not self.hsv_detect:    
            #           img_viz, pmx, pmy, pmw, pmh = self.pm(self.Timage, img, img_viz)
            #           result = self.hsv(img, img_viz, pmx, pmy, pmw, pmh)
            #       else:
            #           result = self.hsv(img, img_viz, -1, -1, -1, -1) # 0531


            #       if len(result) != 0:
            #           Dtarget = result[0]
            #           Dcorners = result[1]  
            #       else:
            #           print("hsv not found")
            #   else :
            #       if self.Target_Stage != "apt":
            #           self.Target_Stage = "apt"
            #           print("apt") 
            #       img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #       result = self.detector.detect(img_gray)
            #       Tresult = []

            #       for r in result:
            #           # 假定result為標記檢測結果，其中包含標記的角點位置信息
            #           points = r.corners
            #           #計算相鄰角點之間的距離
            #           distances = []
            #           for i in range(len(points)):
            #               for j in range(i + 1, len(points)):
            #                   distance = np.linalg.norm(points[i] - points[j])
            #                   distances.append(distance)
            #           distance_mean = np.mean(distances)

            #           # 標記的長寬即為相鄰角點之間的距離的平均值
            #           tag_width = tag_height = distance_mean

            #           if (tag_width > 135 or tag_height > 135) and self.FirstAPT == False: # v4
            #               self.FirstAPT = True
            #               continue
            #           else:
            #               Tresult.append(r)
            #       if len(Tresult) == 0:
            #         result = []

            #       if len(Tresult) != 0:
            #           print(len(result))
            #           Dtarget = result[0].center
            #           Dcorners = result[0].corners
            #           (ptA, ptB, ptC, ptD) = result[0].corners
            #           ptB = (int(ptB[0]), int(ptB[1]))
            #           ptC = (int(ptC[0]), int(ptC[1]))
            #           ptD = (int(ptD[0]), int(ptD[1]))
            #           ptA = (int(ptA[0]), int(ptA[1]))
            #           cv2.line(img_viz, ptA, ptB, (0, 255, 0), 2)
            #           cv2.line(img_viz, ptB, ptC, (0, 255, 0), 2)
            #           cv2.line(img_viz, ptC, ptD, (0, 255, 0), 2)
            #           cv2.line(img_viz, ptD, ptA, (0, 255, 0), 2)
            #       else:
            #           print("apt not found")



            #   if self.FindTag == False:
            #       if len(result) != 0:
            #           self.Detected_Stage = "現在找到"
            #           print("現在找到")
            #           self.FindTag = True
            #           self.target = Dtarget
            #           self.corner = Dcorners
            #           cv2.circle(img_viz, (int(self.target[0]),int(self.target[1])) , 5, (0,0,255), -1)
            #           self.last_img = img.copy()
            #           self.last_alt = self.alt 
            #           self.last_yaw = self.gimbal_yaw
            #       else:
            #           if self.Detected_Stage != "都沒找到":
            #               self.Detected_Stage = "都沒找到"
            #               print("都沒找到")
            #           return       
            #   else:
            #       if len(result) != 0:
            #           if self.Detected_Stage != "持續找到":
            #               self.Detected_Stage = "持續找到"
            #               print("持續找到")
            #           self.target = Dtarget
            #           self.corner = Dcorners
            #           cv2.circle(img_viz, (int(self.target[0]),int(self.target[1])) , 5, (0,0,255), -1)
            #           self.last_img = img.copy()
            #           self.last_alt = self.alt 
            #           self.last_yaw = self.gimbal_yaw
            #       else:
            #           self.Detected_Stage = "現在不見"
            #           print("現在不見")
            #           self.FindTag = False

            #           # 0608
            #           if self.FirstTrack:
            #               img_viz = self.do_fm(img, isFirst = True)
            #           else:
            #               img_viz = self.do_fm(img, isFirst = False)

            #           cv2.circle(img_viz, (int(self.target[0]),int(self.target[1])) , 5, (0,0,255), -1)

            #   t = int(rospy.get_time()) % 2 
            #   cv2.ellipse(img_viz, (int(self.target[0]), int(self.target[1])), (15,15), 0, 10+40*t, 80+40*t, (10,10,255), 3)
            #   cv2.ellipse(img_viz, (int(self.target[0]), int(self.target[1])), (15,15), 0, 100+40*t, 170+40*t, (10,10,255), 3)
            #   cv2.ellipse(img_viz, (int(self.target[0]), int(self.target[1])), (15,15), 0, 190+40*t, 260+40*t, (10,10,255), 3)
            #   cv2.ellipse(img_viz, (int(self.target[0]), int(self.target[1])), (15,15), 0, 280+40*t, 350+40*t, (10,10,255), 3)
            #   dx = int(self.target[0]) - int(self.width/2)

            #   #if self.alt > 20:
            #   # 8cm
            #   d00x,d00y = self.transelt_point(int(self.width/2), int(self.height/2))
            #   d11x,d11y = self.transelt_point(int(self.width/2), int(self.height/2)+5)
            #   dis_ini = math.sqrt((d00x - d11x)**2 + (d00y - d11y)**2) * 100.0
            #   pixel_corr = 5 / dis_ini * 8.0

            #   now_image_height = self.height/2 + pixel_corr
            #   dy = int(self.target[1]) - int(now_image_height)
            #   #elif self.alt <= 20:
            #   #dy = int(self.target[1]) - int(self.height/4*3)

            #   d1x,d1y = self.transelt_point(self.target[0], self.target[1])

            #   #if self.alt > 20:
            #   d2x,d2y = self.transelt_point(int(self.width/2), int(now_image_height))
            #   #elif self.alt <= 20:
            #   #d2x,d2y = self.transelt_point(int(self.width/2), int(self.height/4*3))

            #   d3x,d3y = self.transelt_point(int(self.width/2), self.target[1])

            #   #if self.alt > 20:
            #   d4x,d4y = self.transelt_point(self.target[0], int(now_image_height))
            #   #elif self.alt <= 20:
            #   #  d4x,d4y = self.transelt_point(self.target[0], int(self.height/4*3))

            #   dis = math.sqrt((d2x - d1x)**2 + (d2y - d1y)**2) 
            #   dis1 = math.sqrt((d2x - d3x)**2 + (d2y - d3y)**2) 
            #   dis2 = math.sqrt((d2x - d4x)**2 + (d2y - d4y)**2) 

            #   #if self.alt > 20:
            #   cv2.circle(img_viz, (int(self.width/2),int(now_image_height)), 5, (0,0,0), -1)
            #   #elif self.alt <= 20:
            #   #  cv2.circle(img_viz, (int(self.width/2),int(self.height/4*3)), 5, (0,0,0), -1)

            #   cv2.putText(img_viz, "height: " + str(int(self.alt*1000) / 1000.0)+"m" , (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   #cv2.putText(img_viz, "Find Tag: " + str(self.FindTag), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            #   cv2.putText(img_viz, " x: " + str(dx) + " pixel, " + str(int(dis2*1000)/1000.0) + " m", (12, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   cv2.line(img_viz, (15,30), (11,40), (0, 255, 255), 2)
            #   cv2.line(img_viz, (15,30), (19,40), (0, 255, 255), 2)
            #   cv2.line(img_viz, (11,40), (19,40), (0, 255, 255), 2)

            #   cv2.putText(img_viz, " y: " + str(dy) + " pixel, " + str(int(dis1*1000)/1000.0) + " m", (12, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   cv2.line(img_viz, (15,50), (11,60), (0, 255, 255), 2)
            #   cv2.line(img_viz, (15,50), (19,60), (0, 255, 255), 2)
            #   cv2.line(img_viz, (11,60), (19,60), (0, 255, 255), 2)

            #   cv2.putText(img_viz, " d: " + str(int(dis*1000)/1000.0) + " m", (10, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   cv2.line(img_viz, (15,70), (11,80), (0, 255, 255), 2)
            #   cv2.line(img_viz, (15,70), (19,80), (0, 255, 255), 2)
            #   cv2.line(img_viz, (11,80), (19,80), (0, 255, 255), 2)

            #   #cv2.putText(img_viz, "dis_x: " + str(int(dis2*1000)/1000.0) + " m", (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   #cv2.putText(img_viz, "dis_y: " + str(int(dis1*1000)/1000.0) + " m", (10, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   #if self.alt > 20:
            #   img_viz = self.draw_dotline(img_viz, (int(self.width/2),int(now_image_height)), (int(self.target[0]),int(self.target[1])))
            #   #elif self.alt <= 20:
            #   #  img_viz = self.draw_dotline(img_viz, (int(self.width/2),int(self.height/4*3)), (int(self.target[0]),int(self.target[1])))
            #   #cv2.putText(img_viz, "current angle: " + str(int(self.gimbal_yaw*1000)/1000.0) + " degrees", (10, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   #self.last_frame = img 
            #   cv2.putText(img_viz, "fps: " + str(int(1/(time.time()-stime))) + " frames", (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #   cv2.imshow("img_viz", img_viz)
            #   self.out.write(img_viz)
            #   self.write_data_file.write(str(rospy.get_time())   + "," + str(self.lat) + ","       + str(self.lot) + "," + str(self.alt) + "," +
            #                              str(self.gimbal_yaw)    + "," + str(dis) + ","            + str(dis2) + ","     + str(dis1) + "," +
            #                              str(self.indoor_height) + "," + str(self.velocityZ) + "," + str(self.velocityX) + "," +
            #                              str(self.velocityY)  + "," + str(dx)   + "," + str(dy) + "\n") 
            #   self.point_hl_pub.publish(Float64MultiArray(data = [self.target[0], self.target[1], dis, dis2, dis1, now_image_height ]))
            #   cv2.waitKey(1)

    # def draw_dotline(self, img, pt_s, pt_e):
    #   dis = ((pt_s[0]-pt_e[0])**2+(pt_s[1]-pt_e[1])**2)**.5
    #   gap = 5
    #   pts= []
    #   for i in  np.arange(0, dis, gap):
    #     r = i/dis
    #     x = int((pt_s[0]*(1-r)+pt_e[0]*r)+.5)
    #     y = int((pt_s[1]*(1-r)+pt_e[1]*r)+.5)
    #     p = (x,y)
    #     pts.append(p)
        
    #   s=pts[0]
    #   e=pts[0]
    #   i=0
    #   #print(len(pts))
    #   #t = int(rospy.get_time())
      
    #   for i in range(2, len(pts) - 4):
    #     if i%3==0:
    #       angle = math.atan2(pts[i+1][1] - pts[i][1], pts[i+1][0] - pts[i][0]) + math.pi
    #       x1 = int(pts[i+1][0] + 8 * math.cos(angle - math.pi/4))
    #       y1 = int(pts[i+1][1] + 8 * math.sin(angle - math.pi/4))
    #       x2 = int(pts[i+1][0] + 8 * math.cos(angle + math.pi/4))
    #       y2 = int(pts[i+1][1] + 8 * math.sin(angle + math.pi/4))
          
    #       x3 = x1 - (pts[i+1][0] - pts[i][0]) 
    #       y3 = y1 - (pts[i+1][1] - pts[i][1]) 
    #       x4 = x2 - (pts[i+1][0] - pts[i][0]) 
    #       y4 = y2 - (pts[i+1][1] - pts[i][1]) 
          
    #       t = 3
    #       s = int(rospy.get_time())
    #       if s % 2 == 1:
    #         if i % 2 == 1:
    #           t = 1
    #       elif s % 2 == 0:
    #         if i % 2 == 0:
    #           t = 1
    #       cv2.line(img, (x1,y1), pts[i+1], (0, 0, 252), t)
    #       cv2.line(img, (x2,y2), pts[i+1], (0, 0, 252), t)
    #       cv2.line(img, (x1,y1), (x3,y3), (0, 0, 252), t)
    #       cv2.line(img, (x2,y2), (x4,y4), (0, 0, 252), t)
    #       cv2.line(img, pts[i], (x3,y3), (0, 0, 252), t)
    #       cv2.line(img, pts[i], (x4,y4), (0, 0, 252), t)
          
    #   #cv2.imshow("iiii", img)
    #   #cv2.waitKey(0)
    #   return img
    
    def image_cb(self, ros_data):
        #print(rospy.get_time())
        try: 
            if self.usingUAV == "Bebop":
              self.img = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
            elif self.usingUAV == "DJI":
              np_arr = np.fromstring(ros_data.data, np.uint8)
              img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
              self.img = cv2.resize(img, (int(self.width), int(self.height)), interpolation=cv2.INTER_CUBIC)
              #self.img = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
              #self.pub.publish(self.bridge.cv2_to_imgmsg(self.img, "bgr8"))
        except CvBridgeError as e:
            print(e)
        self.out1.write(self.img)
        
        #231216
        if self.start_take_image == True:
          self.do_alg(self.img)
          
        # if self.start_detect_landmark == True:
        #   self.do_alg(self.img)
        #self.t += 1
        """if self.t == 0:
          self.do_alg(self.img)
          self.t = rospy.get_time()
        elif self.t != 0:
          if rospy.get_time() - self.t >= 1:
            self.do_alg(self.img) 
            self.t = rospy.get_time()"""
            
def main():
    rospy.init_node('collect_data', anonymous=True)
    ss_n = collect_data()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == "__main__":   
    main()  

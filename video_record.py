#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage
import sys
#print(sys.path)
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import glob
import math
import numpy as np
import os
import rospy
import time

from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from distutils.util import strtobool
from numpy.linalg import inv
from pprint import pprint
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray,Header,Empty, String, Float64, Float64MultiArray

# show image for debug
show_image_flag = True
show_mask_flag = True

# rotation matrix
# rot_mtx = np.load('70_mtx.npy')

class collect_data:
  
    def __init__(self):
    
      # image size
      self.width = 428
      self.height = 240
      ###############################################################################
    
      # setting
      self.usingUAV = "DJI"     # or "Bebop"
      self.usingRotView = False # or True, using RotView
      self.usingVTS = True      # or False, using LTS
      self.lookCenter = True    # or False, look top line 
      self.usingGrid = "rec"    # or "rec", using rectangle grid
      
      # image converter
      self.bridge = CvBridge()
      ###############################################################################
      
      # ros sub, pub
      if self.usingUAV == "DJI":
        self.subscriber = rospy.Subscriber("/image/compressed", CompressedImage, self.image_cb,  queue_size = 1, buff_size=2**24)
        self.status_sub = rospy.Subscriber("/dji/status", String, self._status_cb, queue_size = 1) 
      ###############################################################################
      
      # file write
      self.title  = datetime.now().strftime("%Y-%m-%d_%H:%M:%S" + self.usingUAV)
      self.fourcc = cv2.VideoWriter_fourcc('X',"V",'I','D')
      self.out = cv2.VideoWriter(self.title + '.avi', self.fourcc, 20,(self.width, self.height))
        
      self.write_data_file = open(self.title+"_data.csv", 'w')
      self.write_data_file.write("time,lot,lat,alt,down_alt,yaw,camera_pitch,camera_yaw\n")
       
      self.img = np.array([])
       
      #self.now_target = 1
      self.EarthRadius = 6378137
      self.MinLatitude = -85.05112878
      self.MaxLatitude =  85.05112878
      self.MinLongitude = -180
      self.MaxLongitude = 180
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
        self.alt = 0.01
        self.indoor_height = 0.0
        
        self.gimbal_pitch = 0.0  # up 29.2  ~  -90 down
        self.gimbal_yaw = 0.0
        self.gimbal_roll = 0.0
        self.intrinsic_parameters = np.array([639.840550*0.5, 0.000000, 418.805188*0.5, 0.000000, 488.641295*0.5, 237.164378*0.5, 0.000000, 0.000000, 1.000000]).reshape([3,3])
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
      if float(temp_data[14].split("=")[1]) != 0.0:
        self.alt = float(temp_data[14].split("=")[1])
      self.gimbal_pitch = float(temp_data[12].split("=")[1])
      self.gimbal_yaw = float(temp_data[13].split("=")[1])
      self.gimbal_roll = float(temp_data[11].split("=")[1])
      self.indoor_height = float(temp_data[14].split("=")[1])   
        
    def image_cb(self, ros_data):
        #print(rospy.get_time())
        try: 
            if self.usingUAV == "DJI":
              np_arr = np.fromstring(ros_data.data, np.uint8)
              tmp_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
              self.img = cv2.resize(tmp_img, (int(self.width), int(self.height)), interpolation=cv2.INTER_CUBIC)
              self.out.write(self.img)
              if self.indoor_height != 0:
                self.write_data_file.write(str(rospy.get_time()) + "," + str(self.lot) + "," + str(self.lat) + "," + str(self.alt) + "," + str(self.indoor_height) + "," + str(self.yaw) + "," + str(self.gimbal_pitch) + "," + str(self.gimbal_yaw) + "\n")
              else:
                self.write_data_file.write(str(rospy.get_time()) + "," + str(self.lot) + "," + str(self.lat) + "," + str(self.alt) + "," + str(self.indoor_height) + "," + str(self.yaw) + "," + str(self.gimbal_pitch) + "," + str(self.gimbal_yaw) + "\n")
             
        except CvBridgeError as e:
            print(e)
      
def main():
    rospy.init_node('collect_data', anonymous=True)
    ss_n = collect_data()
    #time.sleep(1)
    t = rospy.get_time()
    
    while rospy.get_time() - t < 3:
      pass
    print("OK")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  

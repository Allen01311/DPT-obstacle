# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:27:43 2023

@author: user
"""

import cv2
import geopy.distance
import math
import numpy as np
import pickle
import roslib
import rospy
import space_mod
import sys
import time
import tingyi_DJI_drone

from behave import *
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from pprint import pprint
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, NavSatFix, PointCloud2
from std_msgs.msg import Bool, Empty, Int16MultiArray, Float64MultiArray

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

class btControl_mission:

    # common member  
    # wait_time = 30 # seconds
    # now_time = -1
    
    drone = tingyi_DJI_drone.Drone()
    isContinue = True
    space = space_mod.space_mod()

    #沒差
    width = 856
    height = 480
    
    centerx = 428
    centery = 360

    # bridge = CvBridge() 

    path = [(24.9871405, 121.5734011)]
    end = [-1, -1]
    now_target = 0
    # ini_height = 80.0
    wanted_height = 0
    
    # Take Off
    take0ff_complete = False
    #sp_f = 1.3
    finish = False
    go = 1
    # 0go 往返 回到起點降落
    # 1往 回到起點降落
    
    # memory
    # memory = np.array([])  # momory of pass  
    # REM = np.array([])     # momory of result
    # FixTime = False        # 是否要固定高度進行修正後才下降 0428
    
    stage = 1
    # up stage : 0
    # Search stage : 1
    # Homing stage : 2
    # Down Stage : 3
    # land stage : 4

    def __init__(self):
        
        # self.tree = (
        #     (self.isSearch >> self.Searching) 
        #     | (self.isHoming >> self.Homing)
        #     | (self.isDown >> self.actionDown)
        #     | (self.isLand >> self.landing)
            
        # )

        #起飛+靠近+返回
        # self.tree = (
        #     self.NotFinish >> self.NotFinishTarget >> (self.isNotArrival|self.swichStage ) >> self.fixedPoseAndForward 
        #     | self.ChangeOrLanding
        # )
        
        #起飛+靠近+
        #抵達GPS點+下降(取得indoorheight)+上升+拍照+移動至該邊+
        #返回+著陸
        self.tree = (
            (self.NotFinish >> self.NotFinishTarget >> (self.isNotArrival|self.swichStage ) >> self.fixedPoseAndForward )
            | (self.GPSArrival >> self.down_height >> self.up_height >> self.takeImage >> self.move_Side)
            | (self.isChangeOrLanding >> self.ChangeOrLanding)
            # | (self.isLand >> self.landing)
        )
        
        try:
          with open('btModel_ThreeStage111.pkl', 'rb') as f: 
            model = pickle.load(f)
            #self.loadData(model)
        except:
          print("First run")

        # sub

        #self.point_sub = rospy.Subscriber("/target_point", Int16MultiArray, self.point_cb, queue_size = 1)
    
      

    def saveData(self):
      with open('btModel_ThreeStage.pkl', 'wb') as f:
        dict = {}
        dict["stage"] = btControl_mission.stage
        dict["space"] = btControl_mission.space
        dict["take0ff_complete"] = btControl_mission.take0ff_complete
        print(dict)
        pickle.dump(dict, f)
        #raise KeyboardInterrupt

    def loadData(self, dict):
        btControl_mission.stage = dict["stage"]
        btControl_mission.space = dict["space"]
        btControl_mission.take0ff_complete = dict["take0ff_complete"]

    # @condition
    # def isUp(self):
    #     print("condition: isUp", btControl_mission.stage, btControl_mission.drone.state.battery)
    #     return btControl_mission.stage == 0
        
    # @condition
    # def isHover(self):
    #     print("condition: isHover", btControl_mission.stage, btControl_mission.drone.state.battery)
    #     return btControl_mission.stage == 1
    
    # @condition
    # def isSearch(self):
    #     print("condition: isSearch", btControl_mission.stage, btControl_mission.drone.state.battery)
    #     return btControl_mission.stage == 1
    
    @condition
    def NotFinish(self):
        print("condition: NotFinish", btControl_mission.stage, btControl_mission.drone.state.battery)
        return btControl_mission.stage == 1
      
    @condition
    def NotFinishTarget(self):
        print("condition: NotFinishTarget", btControl_mission.stage, btControl_mission.drone.state.battery)
        return btControl_mission.now_target < len(btControl_mission.path)
        
    
    @condition
    def isNotArrival(self):
        print("condition: isNotArrival")
        print(btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
        if btControl_mission.drone.state.lat != 500 and btControl_mission.drone.state.lot != 500:
          now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
          toT = (btControl_mission.path[btControl_mission.now_target][0], btControl_mission.path[btControl_mission.now_target][1])
          dis = geopy.distance.geodesic(now, toT).m
          print(dis)
          start_time = rospy.get_time()
          while dis <= 3.0 and dis > 2.0:
            t = Twist()
            btControl_mission.drone.flightCrtl.move(t, 0.1)
            now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
            dis = geopy.distance.geodesic(now, toT).m
            if rospy.get_time() - start_time >= 1:
              print("pass 2s and not reach")
              break
          return not (dis <= 2.0)
    
    @condition
    def GPSArrival(self):
        print("condition: GPSArrival", btControl_mission.stage, btControl_mission.drone.state.battery)
        return btControl_mission.stage == 2
    
    @condition
    def isChangeOrLanding(self):
        print("condition: isLand", btControl_mission.stage, btControl_mission.drone.state.battery) 
        return btControl_mission.stage == 3
      
    # @condition
    # def isHoming(self):
    #     print("condition: isHoming", btControl_mission.stage, btControl_mission.drone.state.battery)
    #     return btControl_mission.stage == 2

    # @condition
    # def isDown(self):
    #     print("condition: isDown", btControl_mission.stage, btControl_mission.drone.state.battery)
    #     return btControl_mission.stage == 3
           
    # @condition
    # def isLand(self):
    #     print("condition: isLand", btControl_mission.stage, btControl_mission.drone.state.battery)
    #     return btControl_mission.stage == 4
    '''
    @action
    def actionUp(self):
      print("action: actionUp", btControl_mission.drone.state.hl_tx, btControl_mission.drone.state.hl_ty, btControl_mission.drone.state.hl_dis, btControl_mission.drone.state.hl_cy)
      
      #if btControl_mission.drone.state.alt > 20:
      #  btControl_mission.centery = 240
      #elif btControl_mission.drone.state.alt <= 20:
      #  btControl_mission.centery = 360
      
      if btControl_mission.drone.state.alt > (btControl_mission.wanted_height - 0.1):
        print("switch to Search")
        tx = Twist()
        btControl_mission.drone.flightCrtl.move_s(tx, 0.1)
        btControl_mission.stage = 1
        btControl_mission.drone.now_time = rospy.get_time()
      else:
        tx = Twist()
        if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1:
          if btControl_mission.drone.state.hl_dis <= 0.35:
            print("do not to tune")
            btControl_mission.drone.state.hl_tx = -1
            btControl_mission.drone.state.hl_ty = -1
            btControl_mission.drone.state.hl_dis = -1
          elif btControl_mission.drone.state.hl_dis > 0.35:
            print("tune !!")
            
            dis = btControl_mission.drone.state.hl_dis
            vx = 0.0
            vy = 0.0
            if (btControl_mission.drone.state.hl_tx - btControl_mission.centerx) != 0:
              vx = (btControl_mission.drone.state.hl_tx - btControl_mission.centerx) / abs(btControl_mission.drone.state.hl_tx - btControl_mission.centerx)
            if (btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) != 0:
              vy = -(btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) / abs(btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy)
            
            max_xy = max(btControl_mission.drone.state.hl_dis_x, btControl_mission.drone.state.hl_dis_y)
            
            scale_factor = btControl_mission.drone.state.hl_dis_x
            
            if max_xy == btControl_mission.drone.state.hl_dis_x:
              scale_factor = btControl_mission.drone.state.hl_dis_x
            elif max_xy == btControl_mission.drone.state.hl_dis_y:
              scale_factor = btControl_mission.drone.state.hl_dis_y
            
            command_x = vy
            command_y = vx
            if scale_factor != 0:
              command_x = vy * btControl_mission.drone.state.hl_dis_y / scale_factor
              command_y = vx * btControl_mission.drone.state.hl_dis_x / scale_factor
            
            command_dis = math.sqrt(command_x**2 + command_y**2) 
            scale = dis / command_dis
            tx.linear.x = scale * command_x * 0.85
            tx.linear.y = scale * command_y * 0.85
            tx.linear.z = 0.5
            btControl_mission.drone.flightCrtl.move_s(tx, 2.0)
            
            btControl_mission.drone.state.hl_tx = -1
            btControl_mission.drone.state.hl_ty = -1
            btControl_mission.drone.state.hl_dis = -1   
            btControl_mission.drone.state.hl_dis_x = -1
            btControl_mission.drone.state.hl_dis_y = -1     
            
        elif btControl_mission.drone.state.hl_tx == -1 and btControl_mission.drone.state.hl_ty == -1:  
          print("only up")
          tx.linear.z = 1.0
          btControl_mission.drone.flightCrtl.move(tx, 0.2)
    
         #Hover
    @action
    def actionHover(self):
      print("action: actionHover", btControl_mission.drone.now_time, btControl_mission.drone.state.hl_cy)
     
      #if btControl_mission.drone.state.alt > 20:
      #  btControl_mission.centery = 240
      #elif btControl_mission.drone.state.alt <= 20:
      #  btControl_mission.centery = 360      

      if rospy.get_time() - btControl_mission.drone.now_time >= btControl_mission.wait_time:
        print("arrival time, switch to down")
        btControl_mission.stage = 2
      elif rospy.get_time() - btControl_mission.drone.now_time < btControl_mission.wait_time:
        s = (rospy.get_time() - btControl_mission.drone.now_time) % 60
        m = int((rospy.get_time() - btControl_mission.drone.now_time) / 60)
        print(">>>>>>>>>>>>>>>>>>>>>>")
        print("minutes: ", m)
        print("seconds: ", s)
        print(">>>>>>>>>>>>>>>>>>>>>>")
        print(rospy.get_time() - btControl_mission.drone.now_time)
        tx = Twist()
        if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1:
          if btControl_mission.drone.state.hl_dis <= 0.35:
            print("do not to tune")
            btControl_mission.drone.state.hl_tx = -1
            btControl_mission.drone.state.hl_ty = -1
            btControl_mission.drone.state.hl_dis = -1
          elif btControl_mission.drone.state.hl_dis > 0.35:
            print("tune !!")
            
            dis = btControl_mission.drone.state.hl_dis
            vx = 0.0
            vy = 0.0
            if (btControl_mission.drone.state.hl_tx - btControl_mission.centerx) != 0:
              vx = (btControl_mission.drone.state.hl_tx - btControl_mission.centerx) / abs(btControl_mission.drone.state.hl_tx - btControl_mission.centerx)
            if (btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) != 0:
              vy = -(btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) / abs(btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy)
            
            max_xy = max(btControl_mission.drone.state.hl_dis_x, btControl_mission.drone.state.hl_dis_y)
            
            scale_factor = btControl_mission.drone.state.hl_dis_x
            
            if max_xy == btControl_mission.drone.state.hl_dis_x:
              scale_factor = btControl_mission.drone.state.hl_dis_x
            elif max_xy == btControl_mission.drone.state.hl_dis_y:
              scale_factor = btControl_mission.drone.state.hl_dis_y
            
            command_x = vy
            command_y = vx
            if scale_factor != 0:
              command_x = vy * btControl_mission.drone.state.hl_dis_y / scale_factor
              command_y = vx * btControl_mission.drone.state.hl_dis_x / scale_factor
            #command_x = vy * btControl_mission.drone.state.hl_dis_y / scale_factor
            #command_y = vx * btControl_mission.drone.state.hl_dis_x / scale_factor
            
            command_dis = math.sqrt(command_x**2 + command_y**2) 
            scale = dis / command_dis
            tx.linear.x = scale * command_x * 0.5
            tx.linear.y = scale * command_y * 0.5
            #tx.linear.z = 0.5
            btControl_mission.drone.flightCrtl.move_s(tx, 2.0)
            
            btControl_mission.drone.state.hl_tx = -1
            btControl_mission.drone.state.hl_ty = -1
            btControl_mission.drone.state.hl_dis = -1   
            btControl_mission.drone.state.hl_dis_x = -1
            btControl_mission.drone.state.hl_dis_y = -1           
'''
    
    
    #明憲搜尋原code
    # @action
    # def Searching(self):
    #     print("action: Searching", btControl_mission.drone.state.hl_tx, btControl_mission.drone.state.hl_ty, btControl_mission.drone.state.hl_dis, btControl_mission.drone.state.hl_cy)
    #     if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1:  # HSV偵測傳回來值
    #         print("switch to Homing")
    #         tx = Twist()
    #         btControl_mission.drone.flightCrtl.move_s(tx, 0.1) # 等0.1秒
    #         btControl_mission.stage = 2 # 切換至歸位
    #     else :
    #         tx = Twist()
    #         tx.linear.x = 1.5 # 向前搜尋速度
    #         btControl_mission.drone.flightCrtl.move_s(tx, 1.0) # 執行調整 1 秒
    
    
    #以下恭儀之GPS原code     
    @action
    def swichStage(self):
        print("action: swichStage")
        btControl_mission.now_target += 1
        t = Twist()
        btControl_mission.drone.flightCrtl.move_s(t, 0.1)
    
    @action
    def fixedPoseAndForward(self):
      if btControl_mission.now_target < len(btControl_mission.path):
        print("action: fixedPoseAndForward")
        lat1 = btControl_mission.drone.state.lat
        lat2 = btControl_mission.path[btControl_mission.now_target][0]
        lot1 = btControl_mission.drone.state.lot
        lot2 = btControl_mission.path[btControl_mission.now_target][1]
        to_angle = btControl_mission.space.angleFromCoordinate_correct(lat1,lot1,lat2,lot2)
        if btControl_mission.drone.state.yaw < 0 :
          now_angle = 360 + btControl_mission.drone.state.yaw
        else:  
          now_angle = abs(btControl_mission.drone.state.yaw)
        print(to_angle, now_angle)
                
        d_angle = to_angle - now_angle      
        if abs(d_angle) > 180.0:
          raw_sign = np.sign(d_angle)
          d_angle = ( 360 - abs(d_angle) ) * raw_sign * -1
        
        t = Twist()
        # forward
        # max 5.0/s
        t.linear.x = 2.0
        t.angular.z = math.radians(d_angle) / 2.0
        btControl_mission.drone.flightCrtl.move(t, 0.5)
        btControl_mission.stage = 2
    #以上恭儀之GPS原code
    
    #231216取得indoorheight
    @action
    def down_height(self):
        btControl_mission.wanted_height = btControl_mission.drone.state.alt
        print('下降至有indoorheight')
        tx = Twist()
        while btControl_mission.drone.state.indoor_height > 5 or btControl_mission.drone.state.indoor_height == 0:
          tx.linear.z = -0.1
          btControl_mission.drone.flightCrtl.move(tx, 0.5)
        if 0 < btControl_mission.drone.state.indoor_height < 5:
          tx.linear.z = 0
          btControl_mission.drone.flightCrtl.move(tx, 0.5)
    
    #231216取得indoorheight後進行復位      
    @action
    def up_height(self):
        print("up to " + str(self.wanted_height)) #上升至無人機原高度  
        tx = Twist()
        while btControl_mission.drone.state.alt < btControl_mission.wanted_height:
          tx.linear.z = 0.1
          btControl_mission.drone.flightCrtl.move(tx, 0.5)        
        tx.linear.z = 0
        btControl_mission.drone.flightCrtl.move(tx, 0.5)
        
    #231216開啟影像計算
    @action
    def takeImage(self):
        print('寫影像行為')
        btControl_mission.drone.flightCrtl.active_image(True)
        
    #231216往影像計算的目標點移動
    @action
    def move_Side(self):
      print("action: move_Side", btControl_mission.drone.state.hl_tx, btControl_mission.drone.state.hl_ty, btControl_mission.drone.state.hl_dis,  btControl_mission.drone.state.alt)
      if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1 and btControl_mission.centerx != -1 and btControl_mission.drone.state.hl_cy != -1 and btControl_mission.drone.state.hl_dis != -1:              
        # compute angle
        t = Twist()
        rx = -(btControl_mission.drone.state.hl_tx - btControl_mission.centerx)
        ry = -(btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy)
        
        angle = math.degrees(math.atan2(ry, rx))
        if angle < 0 :
          angle = 360 + angle
          
        anglec = 90
        differ_angle = abs(angle - anglec)
        arcDiffer_angle = 360 - abs(differ_angle)
        min_differ_angle = min(differ_angle, arcDiffer_angle)
        t.angular.z = np.sign(angle - anglec) * math.radians(min_differ_angle) / 1.5
        btControl_mission.drone.flightCrtl.move(t, 0.5)
        
        if btControl_mission.drone.state.hl_dis > 0.5:
          print("tune !!")
          # compute angle 
          t = Twist()
          rx = -(btControl_mission.drone.state.hl_tx - btControl_mission.centerx)
          ry = -(btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy)

          angle = math.degrees(math.atan2(ry, rx))
          if angle < 0 :
            angle = 360 + angle

          anglec = 90
          differ_angle = abs(angle - anglec)
          arcDiffer_angle = 360 - abs(differ_angle)
          min_differ_angle = min(differ_angle, arcDiffer_angle)
          t.angular.z = np.sign(angle - anglec) * math.radians(min_differ_angle) / 1.5
          btControl_mission.drone.flightCrtl.move(t, 0.1)
          
          #往目標點移動
          tx = btControl_mission.drone.correct(btControl_mission.drone.state.hl_dis,#真實移動直線距離
                                               btControl_mission.drone.state.hl_tx, #目標x軸座標點
                                               btControl_mission.centerx,           #畫面中心點
                                               btControl_mission.drone.state.hl_ty, #目標y軸座標點
                                               btControl_mission.drone.state.hl_cy, #目標物中心點y
                                               btControl_mission.drone.state.hl_dis_x,#x軸真實移動距離
                                               btControl_mission.drone.state.hl_dis_y,#y軸真實移動距離
                                               "move_Side")

          btControl_mission.drone.flightCrtl.move_s(tx, 1.0) # 執行調整 1 秒
          btControl_mission.drone.state.hl_tx = -1
          btControl_mission.drone.state.hl_ty = -1
          btControl_mission.drone.state.hl_dis = -1
          btControl_mission.drone.state.hl_dis_x = -1
          btControl_mission.drone.state.hl_dis_y = -1 
        
        else:
          tx = Twist()
          tx.linear.z = 0
          btControl_mission.drone.flightCrtl.move(tx, 0.5)
          btControl_mission.drone.flightCrtl.active_image(False)
          print('action: active_image = False')
          btControl_mission.stage = 3
            
    @action
    def ChangeOrLanding(self):
      btControl_mission.drone.flightCrtl.active_image(False)  #231216
      print("action: ChangeOrLanding", btControl_mission.go)
      if btControl_mission.go == 0:
        print("return")
        btControl_mission.path.reverse()
        btControl_mission.now_target = 1
        btControl_mission.go = 1
      else:
        if btControl_mission.end[0] == -1 and btControl_mission.end[1] == -1:
          btControl_mission.end = [btControl_mission.drone.state.home[0], btControl_mission.drone.state.home[1]]
        
        print("return to start", btControl_mission.end)
        now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
        dis = geopy.distance.geodesic(now, btControl_mission.end).m
        
        #######        N:0
        #######W:270         E:90
        #######       S:180
      
        #while dis > 1.0:
        lat1 = now[0]
        lot1 = now[1]
        lat2 = btControl_mission.end[0]
        lot2 = btControl_mission.end[1]
        
        to_angle = btControl_mission.space.angleFromCoordinate_correct(lat1,lot1,lat2,lot2)
        if btControl_mission.drone.state.yaw < 0 :
          now_angle = 360 + btControl_mission.drone.state.yaw
        else:  
          now_angle = abs(btControl_mission.drone.state.yaw)
        print("to, now", to_angle, now_angle)
         
        # -: left,  +: right
        d_angle = to_angle - now_angle
      
        if abs(d_angle) > 180.0:
          raw_sign = np.sign(d_angle)
          d_angle = ( 360 - abs(d_angle) ) * raw_sign * -1
        
        t = Twist()
        t.angular.z = math.radians(d_angle) / 3.0
        btControl_mission.drone.flightCrtl.move_s(t, 3.2)
      
        while dis > 3.0:
          print("to target dis", dis)
          lat1 = now[0]
          lot1 = now[1]
          lat2 = btControl_mission.end[0]
          lot2 = btControl_mission.end[1]
        
          to_angle = btControl_mission.space.angleFromCoordinate_correct(lat1,lot1,lat2,lot2)
          if btControl_mission.drone.state.yaw < 0 :
            now_angle = 360 + btControl_mission.drone.state.yaw
          else:  
            now_angle = abs(btControl_mission.drone.state.yaw)
          print("to, now", to_angle, now_angle)
        
          # -: left,  +: right
          d_angle = to_angle - now_angle
        
          t = Twist()
          t.linear.x = 3.5
          t.angular.z = math.radians(d_angle) / 2.0
          btControl_mission.drone.flightCrtl.move(t, 0.5)
        
          now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
          dis = geopy.distance.geodesic(now, btControl_mission.end).m
      
        print("arrival END 100m")
        
        t = Twist()
        btControl_mission.drone.flightCrtl.move_s(t, 0.5)
        
        btControl_mission.finish = True
        
    
    #單純著陸行為
    # @action
    # def landing(self):
    #     print("down to " + str(2.0))
    #     while btControl_mission.drone.state.alt > 2.5:
    #       tx = Twist()
    #       tx.linear.z = -1.0
    #       btControl_mission.drone.flightCrtl.move(tx,1)
    #     print("down to " + str(btControl_mission.drone.state.alt))
    #     btControl_mission.drone.flightCrtl.ser_land()
    
    # @action    
    # def Homing(self):
        
    #     print("action: Homing", btControl_mission.drone.state.hl_tx, btControl_mission.drone.state.hl_ty, btControl_mission.drone.state.hl_dis)
        
    #     if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1:
    #         if abs(btControl_mission.drone.state.hl_dis) <= 1.4:  # 距離小於0.35公尺
    #             print("switch to Down")
    #             tx = Twist()
    #             btControl_mission.drone.flightCrtl.move_s(tx, 0.1) # 等0.1秒
    #             btControl_mission.stage = 3 # 切換至下降階段
    #         else:
    #             tx = btControl_mission.drone.correct(btControl_mission.drone.state.hl_dis,
    #                                              btControl_mission.drone.state.hl_tx,
    #                                              btControl_mission.centerx,
    #                                              btControl_mission.drone.state.hl_ty, # 偵測出的y
    #                                              btControl_mission.drone.state.hl_cy, # 中心點y
    #                                              btControl_mission.drone.state.hl_dis_x,
    #                                              btControl_mission.drone.state.hl_dis_y,
    #                                              "Homing")
    #             tx.linear.z = 0 #只做水平對齊  nc                                   
    #             btControl_mission.drone.flightCrtl.move_s(tx, 1.0) # 執行調整 1 秒
    #             btControl_mission.drone.state.hl_tx = -1
    #             btControl_mission.drone.state.hl_ty = -1 
    #             btControl_mission.drone.state.hl_dis = -1   
    #             btControl_mission.drone.state.hl_dis_x = -1
    #             btControl_mission.drone.state.hl_dis_y = -1 
    
    # @action
    # def actionDown(self):

    #   print("action: actionDown", btControl_mission.drone.state.hl_tx, btControl_mission.drone.state.hl_ty, btControl_mission.drone.state.hl_dis,  btControl_mission.drone.state.alt)
      
    #   #if btControl_mission.drone.state.alt > 20:
    #   #  btControl_mission.centery = 240
    #   #elif btControl_mission.drone.state.alt <= 20:
    #   #  btControl_mission.centery = 360

      
           
    #   if btControl_mission.drone.state.alt <= 3.0:  # 高度 < 5m 就進入精準下降階段 0429
    #     print("switch to land")  
    #     tx = Twist()
    #     btControl_mission.drone.flightCrtl.move_s(tx, 0.1)
    #     btControl_mission.stage = 4 
    #     #btControl_mission.drone.now_time = rospy.get_time()
    #   else:
    #     tx = Twist()
    #     if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1 and btControl_mission.drone.state.hl_dis != -1:
    #         if btControl_mission.drone.state.alt > 15:
    #             if abs(btControl_mission.drone.state.hl_dis) <= 1.4:
    #               print("do not to tune")
    #               btControl_mission.drone.state.hl_tx = -1
    #               btControl_mission.drone.state.hl_ty = -1
    #               btControl_mission.drone.state.hl_dis = -1
                  
    #             elif btControl_mission.drone.state.hl_dis > 1.4:
    #               print("tune !!")
    #               tx = btControl_mission.drone.correct(btControl_mission.drone.state.hl_dis,
    #                                                    btControl_mission.drone.state.hl_tx,
    #                                                    btControl_mission.centerx,
    #                                                    btControl_mission.drone.state.hl_ty, # 偵測出的y
    #                                                    btControl_mission.drone.state.hl_cy, # 中心點y
    #                                                    btControl_mission.drone.state.hl_dis_x,
    #                                                    btControl_mission.drone.state.hl_dis_y,
    #                                                    "actionDown")
    #               tx.linear.z = -(btControl_mission.drone.state.alt * 0.06 - btControl_mission.drone.state.hl_dis) # 0426
                  
    #               if tx.linear.z > 0:
    #                   tx.linear.z = -1
                  
    #               btControl_mission.drone.flightCrtl.move_s(tx, 1.0) # 執行調整 1 秒
    #               btControl_mission.drone.state.hl_tx = -1
    #               btControl_mission.drone.state.hl_ty = -1
    #               btControl_mission.drone.state.hl_dis = -1   
    #               btControl_mission.drone.state.hl_dis_x = -1
    #               btControl_mission.drone.state.hl_dis_y = -1 
    #         elif btControl_mission.drone.state.alt <= 15:
    #             if abs(btControl_mission.drone.state.hl_dis) <= 0.35:
    #               print("do not to tune")
    #               btControl_mission.drone.state.hl_tx = -1
    #               btControl_mission.drone.state.hl_ty = -1
    #               btControl_mission.drone.state.hl_dis = -1
                  
    #             elif btControl_mission.drone.state.hl_dis > 0.35:
    #               print("tune !!")
    #               tx = btControl_mission.drone.correct(btControl_mission.drone.state.hl_dis,
    #                                                    btControl_mission.drone.state.hl_tx,
    #                                                    btControl_mission.centerx,
    #                                                    btControl_mission.drone.state.hl_ty, # 偵測出的y
    #                                                    btControl_mission.drone.state.hl_cy, # 中心點y
    #                                                    btControl_mission.drone.state.hl_dis_x,
    #                                                    btControl_mission.drone.state.hl_dis_y,
    #                                                    "actionDown")
    #               tx.linear.z = -(btControl_mission.drone.state.alt * 0.06 - btControl_mission.drone.state.hl_dis) # 0426
    #               if tx.linear.z > 0:
    #                   tx.linear.z = -1
                  
    #               btControl_mission.drone.flightCrtl.move_s(tx, 1.0) # 執行調整 1 秒
    #               btControl_mission.drone.state.hl_tx = -1
    #               btControl_mission.drone.state.hl_ty = -1
    #               btControl_mission.drone.state.hl_dis = -1   
    #               btControl_mission.drone.state.hl_dis_x = -1
    #               btControl_mission.drone.state.hl_dis_y = -1 
          
           
    #     elif btControl_mission.drone.state.hl_tx == -1 and btControl_mission.drone.state.hl_ty == -1:  
    #       print("only down")
    #       tx.linear.z = -1.0
    #       tx.linear.z = -(btControl_mission.drone.state.alt * 0.06) # 0426
    #       btControl_mission.drone.flightCrtl.move(tx, 0.2)    

    # @action
    # def landing(self):
    #     print("action: landing", btControl_mission.drone.state.hl_tx, btControl_mission.drone.state.hl_ty, btControl_mission.drone.state.hl_dis, btControl_mission.drone.state.hl_cy)
        
    #     tx = Twist()
        
    #     if btControl_mission.drone.state.hl_tx != -1 and btControl_mission.drone.state.hl_ty != -1:
    #         print("tune !!")
            
    #         dx = 0.0
    #         dy = 0.0
    #         if (btControl_mission.drone.state.hl_tx - btControl_mission.centerx) != 0:
    #           dx = (btControl_mission.drone.state.hl_tx - btControl_mission.centerx)
    #         if (btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) != 0:
    #           dy = (btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) 
            
    #         # 低空時停止下降&校正 0521
    #         if btControl_mission.drone.state.indoor_height <= 0.6:
    #           tx.linear.z = 0
    #           tx.linear.x = 0
    #           tx.linear.y = 0 
    #           btControl_mission.drone.flightCrtl.move_s(tx, 1)
              
    #           if (btControl_mission.drone.state.hl_tx - btControl_mission.centerx) != 0:
    #             dx = (btControl_mission.drone.state.hl_tx - btControl_mission.centerx)
    #           if (btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy) != 0:
    #             dy = (btControl_mission.drone.state.hl_ty - btControl_mission.drone.state.hl_cy)
                
    #           yRDis = btControl_mission.drone.state.hl_dis_y * np.sign(dy)
    #           xRDis = btControl_mission.drone.state.hl_dis_x * np.sign(dx)
                
    #           tx.linear.x = - yRDis / 2 
    #           tx.linear.y = xRDis / 2
    #           btControl_mission.drone.flightCrtl.move_s(tx, 2)
              
    #           if abs(btControl_mission.drone.state.hl_dis) <= 0.04:
    #               btControl_mission.drone.flightCrtl.ser_land()
              
    #         else:
    #           if abs(dx) > 15:
    #             tx.linear.y = (dx / abs(dx)) * 0.1 # 0.05 0.2
    #           if abs(dy) > 15:
    #             tx.linear.x = -(dy / abs(dy)) * 0.1 # 0.05 0.2
    #             tx.linear.z = 0.0
            
    #           if btControl_mission.drone.state.indoor_height >= 1.5:
    #             tx.linear.z = -0.1
    #           if btControl_mission.drone.state.indoor_height < 1.5: # 低空時下降變慢獲得緩衝 
    #             tx.linear.z = -0.05 
              
    #           if btControl_mission.drone.state.indoor_height < 1: # 低空時候進行更細的校正力度 0426
    #             tx.linear.y = (dx / abs(dx)) * 0.06
    #             tx.linear.x = -(dy / abs(dy)) * 0.06

    #           btControl_mission.drone.flightCrtl.move_s(tx, 0.1)
            
    #           # tx.linear.z = 0
    #           # tx.linear.x = 0
    #           # tx.linear.y = 0 
    #           # btControl_mission.drone.flightCrtl.move_s(tx, 1)
            
            
    #         #if abs(dx) <= 15 and abs(dy) <= 15 and btControl_mission.drone.state.indoor_height <= 0.6: #1修改為0.5
    #         # if abs(btControl_mission.drone.state.hl_dis) <= 0.04 and btControl_mission.drone.state.indoor_height <= 0.6: 
    #         #   print("get, land")
    #         #   btControl_mission.drone.flightCrtl.ser_land()
            
          
      
    # def startDetect_landMark(self):
    #   print("start detect road!!!")
    #   s = True
    #   btControl_mission.drone.flightCrtl.lm_detection(s)
      
    # def stopDetect_landMark(self):
    #   print("stop detect road!!!")
    #   s = False
    #   btControl_mission.drone.flightCrtl.lm_detection(s)

    def up_to(self, height):
      print("up to " + str(height))
      while btControl_mission.drone.state.alt < (height - 1.0):
        tx = Twist()
        tx.linear.z = 4.0
        btControl_mission.drone.flightCrtl.move(tx,0.5)
      print("up to " + str(btControl_mission.drone.state.alt))
      tx = Twist()
      btControl_mission.drone.flightCrtl.move(tx,0.5)
      
    def down_to(self, height):
      print("down to " + str(height))
      while btControl_mission.drone.state.alt > (height-0.5):
        t = Twist()
        t.linear.z = -4.0
        btControl_mission.drone.flightCrtl.move(t, 0.5)
      print("down to " + str(btControl_mission.drone.state.alt))
      t = Twist()
      btControl_mission.drone.flightCrtl.move_s(t,0.5)
    
    def run(self):
        while True:
            if btControl_mission.isContinue == False:
                break
            bb = self.tree.blackboard(1)
            state = bb.tick()
            print("state = " + state + "\n")
            if btControl_mission.drone.isStop == True:
              exec("f = open(\"123.txt\",\'rb\')")           
            while state == RUNNING:
                state = bb.tick()
                print("state = " + state + "\n")
                if btControl_mission.drone.isStop == True:
                  exec("f = open(\"123.txt\",\'rb\')")
            assert state == SUCCESS or state == FAILURE

def main():
    rospy.init_node('btControl_mission', anonymous=True)
    print("start...") 
    btCm_n = btControl_mission()
    
    time.sleep(3)
    
    print("take off...")
    btCm_n.drone.flightCrtl.ser_takeoff()
    
    while btCm_n.drone.state.alt < 1.1:
      pass
      
    btCm_n.up_to(100.0)
    
    #t = rospy.get_time()
    #while rospy.get_time() - t < 1:
    #  pass
    
    print("tune camera angle...")
    while btCm_n.drone.state.gimbal_pitch >= -89.0:#-90.0: 
      btCm_n.drone.flightCrtl.ser_gimbal_90()      
    print("camera angle: ", btCm_n.drone.state.gimbal_pitch)
    
    if btCm_n.end[0] == -1 and btCm_n.end[1] == -1:
      while btCm_n.drone.state.home[0] == 500.0 and btCm_n.drone.state.home[1] == 500.0:
        pass
      btCm_n.end = [btCm_n.drone.state.home[0], btCm_n.drone.state.home[1]]   
    
    #t = rospy.get_time()
    #while rospy.get_time() - t < 1:
    #  pass
    #time.sleep(3)
    #btCm_n.startDetect_landMark()
    #btCm_n.takeImage()
    try:
      btCm_n.run()
      print("finish...")
      
      btCm_n.down_to(4.0)
      
      print("landing...")
      btCm_n.drone.flightCrtl.ser_land()
      
      print("OK.")
      rospy.spin()
      
    except Exception as e:
      print("Shutting down ROS Image feature detector module")
      btCm_n.saveData()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

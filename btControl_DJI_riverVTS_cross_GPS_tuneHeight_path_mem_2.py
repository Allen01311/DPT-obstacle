#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import geopy.distance
import math
import numpy as np
import roslib
import rospy
import pickle
import sys
import time
import tingyi_DJI_drone
import space_mod

from behave import *
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from pprint import pprint
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, NavSatFix, PointCloud2
from std_msgs.msg import Bool, Empty


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

class btControl_mission:

    # common member
    drone = tingyi_DJI_drone.Drone()
    isContinue = True
    space = space_mod.space_mod()

    width = 428
    height = 240
    #title = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #allout = cv2.VideoWriter(title + '_all.avi', fourcc, 30,(856, 480))
    bridge = CvBridge() 
    ini_height = 50.0 #22.0
    now_target  = 1
    
    # jingmei long test
    target = [(24.987601049066853, 121.5724122080593), (24.986768959016764, 121.57258033331823), (24.984838590870364, 121.5717863610597), (24.982627421059515, 121.57161473979394)]
    ret_target = [(24.982627421059515, 121.57161473979394), (24.984838590870364, 121.5717863610597), (24.986768959016764, 121.57258033331823),(24.987601049066853, 121.5724122080593)]
    
    # jingmei test
    #target = [(24.986534, 121.572577), (24.985965, 121.572319), (24.984883, 121.571807), (24.983141, 121.571643)]
    #ret_target = [(24.983141, 121.571643), (24.984883, 121.571807), (24.985965, 121.572319), (24.986534, 121.572577)]
    
    # 3.5km to xin dian
    #target = [(24.982627421059515, 121.57161473979394), (24.981937547983645, 121.57112018926544), (24.980400667481536, 121.56741512117718), (24.98103927580985, 121.56503478845475), (24.979500487929663, 121.5614975573203), (24.97985871926997, 121.55866065829841), (24.978781733393404, 121.5577634758118), (24.977451635506828, 121.55754511338104)]
    #ret_target = [(24.977451635506828, 121.55754511338104), (24.978781733393404, 121.5577634758118), (24.97985871926997, 121.55866065829841), (24.979500487929663, 121.5614975573203), (24.98103927580985, 121.56503478845475), (24.980400667481536, 121.56741512117718), (24.981937547983645, 121.57112018926544), (24.982627421059515, 121.57161473979394)]
    
    # 3.5km to zoo
    #target = [(24.980745250372138, 121.5684162240155), (24.981937547983645, 121.57112018926544), (24.982627421059515, 121.57161473979394), (24.984838590870364, 121.5717863610597), (24.986768959016764, 121.57258033331823),(24.98894792857531, 121.57179526861287),(24.994559069130663, 121.5723773431982)]
    #ret_target = [(24.994559069130663, 121.5723773431982), (24.98894792857531, 121.57179526861287), (24.986768959016764, 121.57258033331823), (24.984838590870364, 121.5717863610597), (24.982627421059515, 121.57161473979394), (24.981937547983645, 121.57112018926544), (24.980745250372138, 121.5684162240155)]
    
    # 2km to U
    #target = [(24.979364, 121.553440), (24.976924, 121.554129), (24.976350, 121.556055), (24.976768, 121.557547), (24.979228, 121.557902),(24.980065, 121.559019),(24.979496, 121.561295),(24.981155, 121.565254),(24.980560, 121.567620),(24.981050, 121.568993), (24.981807, 121.569117)]
    #ret_target = [(24.979364, 121.553440), (24.976924, 121.554129), (24.976350, 121.556055), (24.976768, 121.557547), (24.979228, 121.557902),(24.980065, 121.559019),(24.979496, 121.561295),(24.981155, 121.565254),(24.980560, 121.567620),(24.981050, 121.568993), (24.981807, 121.569117)]
    
    # miao factory river point
    #target = [(24.599144, 120.815315), (24.599252, 120.817801), (24.598774, 120.821406), (24.597405, 120.822343)]
    #ret_target = [(24.597405, 120.822343), (24.598774, 120.821406), (24.599252, 120.817801), (24.599144, 120.815315)]
    
    # jingmei tiny river test
    #target = [(24.982912, 121.572473), (24.983409, 121.573185), (24.983984, 121.573960)]
    #ret_target = [(24.983984, 121.573960), (24.983409, 121.573185), (24.982912, 121.572473)]
    
    # tainan 2ren
    #target = [(22.889805, 120.330009), (22.889445, 120.331150), (22.889349, 120.333152), (22.889387, 120.335631), (22.889395, 120.337247)]
    #ret_target = [(22.889395, 120.337247), (22.889387, 120.335631), (22.889349, 120.333152), (22.889445, 120.331150), (22.889805, 120.330009)]
    
    #start = (24.982627421059515, 121.57161473979394) #24.983337395672795, 121.57229069743312
    #end = (24.981807, 121.569117)
    end = [-1, -1]
    
    # Take Off
    take0ff_complete = False
    
    # bt finish condition
    finish = False
    
    # using rot view
    rot = False
    
    # last moving direction
    last_direction = -1
    
    # segm data is received
    isReceived = False
    
    # first received
    first_receive = False
    
    # current time
    current = -1
    
    # go (0) and return(1)
    go = 0

    # needTune Height numbers
    needTune_n = 0
    
    # record moving time
    need_time = -1
    last_need = -1
    
    # compensate_height
    compensate_height = 3
    mem_list = []
    
    def __init__(self): #self.tree = (( self.isPointUpdate >> self.flyingControl | self.slowFlying))
        self.tree = (
            self.NotFinish >> self.runRiverModel >> ((self.needTuneHeight >> self.tuneHeight) | self.move4point)
            |self.ChangeOrLand
        )

        try:
          with open('btModel_BS.pkl', 'rb') as f: 
            model = pickle.load(f)
            self.loadData(model)
        except:
          print("First run")
        #self.title = '2020-02-04-DJI_rivertest'
        #self.fourcc = cv2.VideoWriter_fourcc('X',"V",'I','D')
        #self.out = cv2.VideoWriter(self.title + '_video.avi', self.fourcc, 30,(840, 490))
        # sub
        #self.subscriber = rospy.Subscriber("/image/compressed", CompressedImage, self.image_cb,  queue_size = 1, buff_size=2**24)
        #self.pc2_sub = rospy.Subscriber("/orb_slam2_mono/map_points", PointCloud2, self.pc2_cb, queue_size = 1)

    def saveData(self):
      with open('btModel_BS.pkl', 'wb') as f:
        dict = {}
        dict["space"] = btControl_mission.space
        """dict["take0ff_complete"] = btControl_mission.take0ff_complete
        dict["ini_height"] = btControl_mission.ini_height
        dict["ini_width"] = btControl_mission.ini_width
        dict["downH"] = btControl_mission.downH
        dict["direction"] = btControl_mission.direction
        dict["isSetHome"] = btControl_mission.drone.isSetHome
        dict["home"] = btControl_mission.drone.state.home
        dict["isSetYaw"] = btControl_mission.drone.isSetYaw
        dict["iniPose"] = btControl_mission.drone.state.iniPose
        dict["now_pos"] = btControl_mission.now_pos
        dict["now_h"] = btControl_mission.now_h
        dict["dire"] = btControl_mission.dire
        dict["turn_angle"] = btControl_mission.turn_angle
        dict["now_dis"] = btControl_mission.now_dis"""
        print(dict)
        pickle.dump(dict, f)
        #raise KeyboardInterrupt

    def loadData(self, dict):
        btControl_mission.space = dict["space"]
        """btControl_mission.take0ff_complete = dict["take0ff_complete"]
        btControl_mission.ini_height = dict["ini_height"]
        btControl_mission.ini_width = dict["ini_width"]
        btControl_mission.downH = dict["downH"]
        btControl_mission.direction = dict["direction"]
        btControl_mission.drone.isSetHome = dict["isSetHome"]
        btControl_mission.drone.state.home = dict["home"]
        btControl_mission.drone.isSetYaw = dict["isSetYaw"]
        btControl_mission.drone.state.iniPose = dict["iniPose"]
        btControl_mission.now_pos = dict["now_pos"]
        btControl_mission.now_h = dict["now_h"]
        btControl_mission.dire = dict["dire"]
        btControl_mission.turn_angle = dict["turn_angle"]
        btControl_mission.now_dis = dict["now_dis"] """

        
    @condition
    def isPointUpdate(self):
      print("condition: isPointUpdate")
      return btControl_mission.drone.state.index != 4

    @action
    def flyingControl(self):
      print("action: flyingControl")
      if btControl_mission.drone.state.index == 1:
          dx = (btControl_mission.drone.state.grid_x - 214)
          dy = (btControl_mission.drone.state.tp_y - 120)
          
          rx = -(btControl_mission.drone.state.tp_x - btControl_mission.drone.state.grid_x)
          ry = -(btControl_mission.drone.state.tp_y - btControl_mission.drone.state.grid_y)
              
          angle = math.degrees(math.atan2(ry, rx))
          if angle < 0 :
            angle = 360 + angle

          anglec = 90
            
          t = Twist()
          
          differ_metric = 10
          angle_metric = 2
          if btControl_mission.rot == False:
            differ_metric = 10
            angle_metric = 2
          elif btControl_mission.rot == True:
            differ_metric = 10
            angle_metric = 2
          
          if abs(dx) > differ_metric:
            t.linear.y = dx / abs(dx) * 0.3
            
          t.linear.x = 1.0
          #print(angle,anglec)
          
          
          if angle > 90 + angle_metric:
              t.angular.z = -((abs(angle - anglec) / 1.8) * math.pi) / 180.0
          elif 90 - angle_metric > angle:
              t.angular.z = ((abs(angle - anglec) / 1.8) * math.pi) / 180.0
          
          #if int(btControl_mission.drone.state.alt) != 25:
          #  if int(btControl_mission.drone.state.alt) > 25:
          #    t.linear.z = -0.2
          #  elif int(btControl_mission.drone.state.alt) < 25:
          #    t.linear.z = 0.2
          btControl_mission.drone.flightCrtl.move(t, 1.5)
          btControl_mission.last_direction = btControl_mission.drone.state.index
          btControl_mission.drone.state.index = 4

    @action
    def slowFlying(self):
      print("action: slowFlying")
      t = Twist()
      t.linear.x = 0.8
      btControl_mission.drone.flightCrtl.move(t, 0.5)

    @condition
    def NotFinish(self):
    
      if btControl_mission.finish == False:
    
        print("condition: NotFinish")
        print(btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
        if btControl_mission.drone.state.lat != 500 and btControl_mission.drone.state.lot != 500:
          #if btControl_mission.end[0] == -1 and btControl_mission.end[0] == -1:
          #  btCm_n.end = [btControl_mission.drone.state.lat, btControl_mission.drone.state.lot]
          
          now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
          toT = (btControl_mission.target[-1][0], btControl_mission.target[-1][1])
          if btControl_mission.go == 1:
            toT = (btControl_mission.ret_target[-1][0], btControl_mission.ret_target[-1][1])
          dis = geopy.distance.vincenty(now, toT).m
          print(dis)
          
          start_time = rospy.get_time()
          while dis <= 5.2 and dis > 5.0:
            t = Twist()
            btControl_mission.drone.flightCrtl.move(t, 0.2)
            now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
            dis = geopy.distance.vincenty(now, toT).m
            if rospy.get_time() - start_time >= 3:
              print("pass 3s and not reach")
              break
          return not (dis < 5.0)

    @action
    def runRiverModel(self):
      print("action: runRiverModel")
      if btControl_mission.isReceived == False:
        btControl_mission.drone.flightCrtl.active_river()
        btControl_mission.isReceived = True
        btControl_mission.current = rospy.get_time() 

    @condition
    def needTuneHeight(self):
      print("condition: needTuneHeight", btControl_mission.drone.state.river_width)
      if btControl_mission.drone.state.river_width != -1:
        btControl_mission.mem_list.append(btControl_mission.drone.state.river_width)  
        btControl_mission.drone.state.river_width = -1
      
      if btControl_mission.drone.state.indoor_height != 0:
        alt = btControl_mission.drone.state.alt
      else:
        alt = btControl_mission.drone.state.alt + btControl_mission.compensate_height
      
      if alt < 12:
        return True
        
      if len(btControl_mission.mem_list) > 2:
        print(btControl_mission.mem_list[-3:])
        temp = np.array(btControl_mission.mem_list[-3:])
        if 0.40 > btControl_mission.mem_list[-1] > 0.2:
          return False
        
        if len(temp[temp==0]) >= 2:  
          return True
        elif len(temp[(temp > 0.0) & (temp < 0.2)]) >= 2 and alt >= 13:
          return True
        elif len(temp[temp > 0.40]) >= 2:
          return True
        else:
          return False
        
      else:
        return False
      
      #if btControl_mission.drone.state.river_width == 0:
      #  btControl_mission.needTune_n += 1
      #  btControl_mission.drone.state.river_width = -1
      #  return btControl_mission.needTune_n % 3 == 0
      #else:
      #  if btControl_mission.drone.state.river_width != -1:
      #    btControl_mission.needTune_n = 0
      #  return (0.10 < btControl_mission.drone.state.river_width < 0.18) or (0.5 < btControl_mission.drone.state.river_width)
      
    @action
    def tuneHeight(self):
      print("action: tuneHeight", btControl_mission.drone.state.river_width)
      t = Twist()
      
      if btControl_mission.drone.state.indoor_height != 0:
        alt = btControl_mission.drone.state.alt
      else:
        alt = btControl_mission.drone.state.alt + btControl_mission.compensate_height
      
      if alt < 12:
        t.linear.z = 1.0
        btControl_mission.drone.flightCrtl.move(t, 0.1)  
        return FAILURE
        
      temp = np.array(btControl_mission.mem_list[-3:])
      if len(temp[temp==0]) >= 2: 
        
        if alt <= 12:
          t.linear.z = 1.0
        elif alt >= 17:
          t.linear.z = -1.0 
        
        btControl_mission.drone.flightCrtl.move_s(t, 3.0)  
        btControl_mission.mem_list.pop()
        btControl_mission.isReceived = False
        return FAILURE
      else: 
        if len(temp[(temp > 0.0) & (temp < 0.2)]) >= 2 and alt >= 13:
          t.linear.z = -1.0
        elif len(temp[temp > 0.40]) >= 2:
          t.linear.z = 1.0
        
        btControl_mission.drone.flightCrtl.move(t, 0.1)  
      
        if rospy.get_time() - btControl_mission.current >= 1.0 and btControl_mission.current != -1:
          btControl_mission.isReceived = False
      
    @action
    def move4point(self):
        
        print("action: move4GPSpoint, target: ", btControl_mission.drone.state.rtx, btControl_mission.drone.state.rty)
        print("action: move4GPSpoint, waypoint: ",  btControl_mission.drone.state.rgx, btControl_mission.drone.state.rgy) 
        print("action: move4GPSpoint, center: ", btControl_mission.drone.state.rcx, btControl_mission.drone.state.rcy)
        print("action: move4GPSpoint, onRiver?: ", btControl_mission.drone.state.is_on_river)
        print("action: move4GPSpoint, distance: ", btControl_mission.drone.state.c2t_dis)
        print("action: move4GPSpoint, hor, ver: ", btControl_mission.drone.state.c2t_hor_dis, btControl_mission.drone.state.c2t_ver_dis)
        
        # inintial 
        if btControl_mission.drone.state.rtx == -1 and btControl_mission.drone.state.rty == -1 and btControl_mission.drone.state.rgx == -1 and btControl_mission.drone.state.rgy == -1 and btControl_mission.drone.state.rcx == -1 and btControl_mission.drone.state.rcy == -1:
          
          print("not receive...")
          if rospy.get_time() - btControl_mission.current > 2.0 and btControl_mission.current != -1:
            # restart river segm
            btControl_mission.isReceived = False
        
        # loop 
        else:
          # receive point
          if btControl_mission.drone.state.rtx != -1 and btControl_mission.drone.state.rty != -1 and btControl_mission.drone.state.rcx != -1 and btControl_mission.drone.state.rcy != -1:
            
            # on river
            if btControl_mission.drone.state.is_on_river == 1:
              
              t = Twist()
              
              # compute angle
              rx = -(btControl_mission.drone.state.rtx - btControl_mission.drone.state.rcx)
              ry = -(btControl_mission.drone.state.rty - btControl_mission.drone.state.rcy)
              
              angle = math.degrees(math.atan2(ry, rx))
              if angle < 0 :
                angle = 360 + angle

              anglec = 90 
              differ_angle = abs(angle - anglec)
              arcDiffer_angle = 360 - abs(differ_angle)
              min_differ_angle = min(differ_angle, arcDiffer_angle)

              btControl_mission.need_time = btControl_mission.drone.state.c2t_dis / 2.5
              btControl_mission.last_need = rospy.get_time()
                            
              if btControl_mission.need_time > 2.0:
              
                t.linear.x = 2.5
                t.angular.z = np.sign(angle - anglec) * math.radians(min_differ_angle) / 1.5
                btControl_mission.drone.flightCrtl.move(t, 0.5)
              
                while rospy.get_time() - btControl_mission.last_need < 0.9:
                  pass
              
                btControl_mission.need_time -= 1.5
                
                t = Twist()
                btControl_mission.last_need = rospy.get_time()
              
                t.linear.x = 2.5
                btControl_mission.drone.flightCrtl.move(t, btControl_mission.need_time / 4.0)
              else:
                print("less than 5m dis")
                btControl_mission.need_time = 2.5
                btControl_mission.last_need = rospy.get_time()
                s = btControl_mission.drone.state.c2t_dis / 2.5
                print(s)
                t.linear.x = s
                t.angular.z = np.sign(angle - anglec) * math.radians(min_differ_angle) / 1.5
                btControl_mission.drone.flightCrtl.move(t, 0.5)
              
                while rospy.get_time() - btControl_mission.last_need < 0.9:
                  pass
              
                btControl_mission.need_time -= 1.5
                
                t = Twist()
                btControl_mission.last_need = rospy.get_time()
              
                t.linear.x = s
                btControl_mission.drone.flightCrtl.move(t, btControl_mission.need_time / 4.0)
              
              #if int(btControl_mission.drone.state.alt) != 25:
              #  if int(btControl_mission.drone.state.alt) > 25:
              #    t.linear.z = -0.5
              #  elif int(btControl_mission.drone.state.alt) < 25:
              #    t.linear.z = 0.5  
              
              btControl_mission.drone.state.rtx = -1  
              btControl_mission.drone.state.rty = -1  
              btControl_mission.drone.state.rcx = -1 
              btControl_mission.drone.state.rcy = -1
              if btControl_mission.first_receive == False:
                btControl_mission.first_receive = True
              btControl_mission.isReceived = False
              
            # not on river
            elif btControl_mission.drone.state.is_on_river == 0:  
              t = Twist()
              
              dis = btControl_mission.drone.state.c2t_dis
              hor_dis = btControl_mission.drone.state.c2t_hor_dis
              ver_dis = btControl_mission.drone.state.c2t_ver_dis
              
              vx = 0.0
              vy = 0.0
              sign_x = np.sign(btControl_mission.drone.state.rtx - btControl_mission.drone.state.rcx)
              sign_y = np.sign(btControl_mission.drone.state.rty - btControl_mission.drone.state.rcy)
              
              max_xy = max(hor_dis, ver_dis)
              scale_factor = hor_dis
              
              if max_xy == ver_dis:
                scale_factor = ver_dis 
            
              command_x = vy
              command_y = vx
              
              if scale_factor != 0:
                command_x = -1 * sign_y * ver_dis / scale_factor
                command_y = sign_x * hor_dis / scale_factor
              
              command_dis = math.sqrt(command_x ** 2 + command_y ** 2)
              #scale = dis / command_dis
              
              t.linear.x = command_x * 2.
              t.linear.y = command_y * 2.
              
              btControl_mission.need_time = dis / 2.5
              btControl_mission.last_need = rospy.get_time()
              
              btControl_mission.drone.flightCrtl.move(t, btControl_mission.need_time / 4.0)
              
              btControl_mission.drone.state.rtx = -1  
              btControl_mission.drone.state.rty = -1  
              btControl_mission.drone.state.rcx = -1 
              btControl_mission.drone.state.rcy = -1
              if btControl_mission.first_receive == False:
                btControl_mission.first_receive = True
              btControl_mission.isReceived = False
          
          else:
            
            print("time: ", btControl_mission.need_time, btControl_mission.last_need)
            if rospy.get_time() - btControl_mission.last_need < btControl_mission.need_time:
              print("continue")
            
            else:
              print("pass point, using GPS")
              t = Twist()
              
              now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
              t_gps = (btControl_mission.drone.state.rgx, btControl_mission.drone.state.rgy)
              
              # equal
              if btControl_mission.target[btControl_mission.now_target][0] == btControl_mission.drone.state.rgx and btControl_mission.target[btControl_mission.now_target][1] == btControl_mission.drone.state.rgy:
                dis = geopy.distance.vincenty(now, t_gps).m
                
                # not near
                if dis >= 30:
                  print("not near")
                # near, but inspection not change 
                else:
                  btControl_mission.now_target = np.clip(btControl_mission.now_target + 1, 0, len(btControl_mission.target) - 1 )
                  t_gps = (btControl_mission.target[btControl_mission.now_target][0], btControl_mission.target[btControl_mission.now_target][1])
              # not equal    
              else:
                
                if btControl_mission.now_target + 1 < len(btControl_mission.target):
                  if btControl_mission.target[btControl_mission.now_target+1][0] == btControl_mission.drone.state.rgx and btControl_mission.target[btControl_mission.now_target+1][1] == btControl_mission.drone.state.rgy:
                    print("change control index")
                    btControl_mission.now_target = np.clip(btControl_mission.now_target + 1, 0, len(btControl_mission.target) - 1 )
                    t_gps = (btControl_mission.target[btControl_mission.now_target][0], btControl_mission.target[btControl_mission.now_target][1])
                  else: 
                    print("near target and control change fastly") 
                else:
                  print("last point")
                  btControl_mission.now_target = len(btControl_mission.target) - 1
                  t_gps = (btControl_mission.target[btControl_mission.now_target][0], btControl_mission.target[btControl_mission.now_target][1])   
                                 
              print(now, t_gps)
              to_angle = btControl_mission.space.angleFromCoordinate_correct(now[0],now[1],t_gps[0],t_gps[1])
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
              t.linear.x = 2.5
              #if abs(d_angle) < 90:
              if abs(d_angle) > 45:
                t.angular.z = math.radians(d_angle) / 2.0
                btControl_mission.drone.flightCrtl.move_s(t, 2.0)
              else:
                t.angular.z = math.radians(d_angle) #/ 2.0
                btControl_mission.drone.flightCrtl.move_s(t, 1.0)
              
              if rospy.get_time() - btControl_mission.current > 5.0 and btControl_mission.current != -1:
                btControl_mission.isReceived = False
            
              if btControl_mission.first_receive == True:
                t = Twist()
                t.linear.x = 2.5
                btControl_mission.drone.flightCrtl.move(t, 0.1) 
        
        
        # check GPS is need to update
        now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
        t_gps = (btControl_mission.drone.state.rgx, btControl_mission.drone.state.rgy)
              
        # equal
        if btControl_mission.target[btControl_mission.now_target][0] == btControl_mission.drone.state.rgx and btControl_mission.target[btControl_mission.now_target][1] == btControl_mission.drone.state.rgy:
          dis = geopy.distance.vincenty(now, t_gps).m
                
          # not near
          if dis >= 30:
            print("not near")
          # near, but inspection not change 
          else:
            btControl_mission.now_target = np.clip(btControl_mission.now_target + 1, 0, len(btControl_mission.target) - 1 )
        # not equal    
        else:
                
          if btControl_mission.now_target + 1 < len(btControl_mission.target):
            if btControl_mission.target[btControl_mission.now_target+1][0] == btControl_mission.drone.state.rgx and btControl_mission.target[btControl_mission.now_target+1][1] == btControl_mission.drone.state.rgy:
              print("change control index")
              btControl_mission.now_target = np.clip(btControl_mission.now_target + 1, 0, len(btControl_mission.target) - 1 )
            else: 
              print("near target and control change fastly") 
          else:
            print("last point")
            btControl_mission.now_target = len(btControl_mission.target) - 1
              
        now_time = rospy.get_time()    
        while rospy.get_time() - now_time < 0.1:
          pass
        
    @action
    def ChangeOrLand(self):
      print("action: ChangeOrLand")
      
      t = Twist()
      btControl_mission.drone.flightCrtl.move_s(t, 0.5) 
      
      if btControl_mission.go == 0:
        print("update return pathing...")
        btControl_mission.drone.flightCrtl.active_return()
        
        print("turn to target")
        now_time = rospy.get_time()
        while rospy.get_time() - now_time < 1.0:
          pass
      
        print("turning...")
        to_angle = btControl_mission.space.angleFromCoordinate_correct(btControl_mission.drone.state.lat, btControl_mission.drone.state.lot, btControl_mission.target[-2][0],btControl_mission.target[-2][1])
      
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
          
        print("finish to return_to_start")   
        t = rospy.get_time()
        while  rospy.get_time() - t < 3.0:
          pass
        
        print("update return pathing...")
        btControl_mission.drone.flightCrtl.active_return()
        
        t = rospy.get_time()
        while  rospy.get_time() - t < 1.0:
          pass
        
        btControl_mission.drone.state.rtx = -1
        btControl_mission.drone.state.rty = -1  
        btControl_mission.drone.state.rgx = -1
        btControl_mission.drone.state.rgy = -1 
        btControl_mission.drone.state.rcx = -1
        btControl_mission.drone.state.rcy = -1 
        btControl_mission.isReceived = False
        btControl_mission.first_receive = False
        btControl_mission.current = -1
        btControl_mission.now_target = 1
        btControl_mission.target = btControl_mission.ret_target
        btControl_mission.go = 1
        #btControl_mission.finish = True
        #print("action: land(fake because may be drop river...)")
      
      elif btControl_mission.go == 1:
      
        if btControl_mission.end[0] == -1 and btControl_mission.end[1] == -1:
          btControl_mission.end = [btControl_mission.drone.state.home[0], btControl_mission.drone.state.home[1]]
        
        print("return to end", btControl_mission.end)
        now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
        #dis = geopy.distance.vincenty(now, btControl_mission.drone.state.home).m
        dis = geopy.distance.vincenty(now, btControl_mission.end).m
        
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
          dis = geopy.distance.vincenty(now, btControl_mission.end).m
      
        print("arrival END 50m")
        
        t = Twist()
        btControl_mission.drone.flightCrtl.move_s(t, 0.5) 
        
        btControl_mission.finish = True
           
    def up_to(self, height):
      print("up to " + str(height))
      while btControl_mission.drone.state.alt < (height-1.0):
        t = Twist()
        t.linear.z = 4.0
        btControl_mission.drone.flightCrtl.move(t, 0.5)
      print("up to " + str(btControl_mission.drone.state.alt))
      t = Twist()
      btControl_mission.drone.flightCrtl.move_s(t,0.5)

    def down_to(self, height):
      print("down to " + str(height))
      while btControl_mission.drone.state.alt > (height-0.5):
        t = Twist()
        t.linear.z = -4.0
        btControl_mission.drone.flightCrtl.move(t, 0.5)
      print("down to " + str(btControl_mission.drone.state.alt))
      t = Twist()
      btControl_mission.drone.flightCrtl.move_s(t,0.5)
 
    def go_to_start(self):
      print("start point: ", btControl_mission.target[0])
      now = (btControl_mission.drone.state.lat, btControl_mission.drone.state.lot)
      dis = geopy.distance.vincenty(now, btControl_mission.target[0]).m
      
      #######        N:0
      #######W:270         E:90
      #######       S:180
      
      #while dis > 1.0:
      lat1 = now[0]
      lot1 = now[1]
      lat2 = btControl_mission.target[0][0]
      lot2 = btControl_mission.target[0][1]
        
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
        lat2 = btControl_mission.target[0][0]
        lot2 = btControl_mission.target[0][1]
        
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
        dis = geopy.distance.vincenty(now, btControl_mission.target[0]).m
      
      t = Twist()
      btControl_mission.drone.flightCrtl.move_s(t, 0.5) 
        
      print("turn to target")
      now_time = rospy.get_time()
      while rospy.get_time() - now_time < 2.0:
        pass
      
      
      print("turning...")
      to_angle = btControl_mission.space.angleFromCoordinate_correct(btControl_mission.drone.state.lat, btControl_mission.drone.state.lot, btControl_mission.target[1][0],btControl_mission.target[1][1])
      
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
          
      print("finish to go_to_start")   
      t = rospy.get_time()
      while  rospy.get_time() - t < 3.0:
        pass
      
 
    def run(self):
        while True:
            if btControl_mission.finish == True:
                break
            bb = self.tree.blackboard(1)
            state = bb.tick()
            print "state = %s\n" % state
            if btControl_mission.drone.isStop == True:
              exec("f = open(\"123.txt\",\'rb\')")           
            while state == RUNNING:
                state = bb.tick()
                print "state = %s\n" % state
                if btControl_mission.drone.isStop == True:
                  exec("f = open(\"123.txt\",\'rb\')")
            assert state == SUCCESS or state == FAILURE
            t = rospy.get_time()
            while rospy.get_time() - t < 1.0:
              pass

def main():
    rospy.init_node('btControl_mission_flight', anonymous=True)
    print("start...") 
    btCm_n = btControl_mission()
    #btCm_n.tuneCamera()
    #btCm_n.takeoff()
    time.sleep(3)

    print("take off...")
    btCm_n.drone.flightCrtl.ser_takeoff()
    
    while btCm_n.drone.state.alt < 1.1:
      pass
      
    btCm_n.up_to(50.0)
    #btCm_n.drone.flightCrtl.ser_land()    
    
    print("tune camera angle...")
    while btCm_n.drone.state.gimbal_pitch > -89:#-90.0: 
      btCm_n.drone.flightCrtl.ser_gimbal_90()  
    print("camera angle: ", btCm_n.drone.state.gimbal_pitch)
    
      
    print("go to start...")
    btCm_n.go_to_start()
    
    if btCm_n.end[0] == -1 and btCm_n.end[1] == -1:
      while btCm_n.drone.state.home[0] == 500.0 and btCm_n.drone.state.home[1] == 500.0:
        pass
      btCm_n.end = [btCm_n.drone.state.home[0], btCm_n.drone.state.home[1]]
        
    try:
        btCm_n.run()
        print("finish...")
        
        btCm_n.down_to(4.0)
        
        print("landing...")
        btCm_n.drone.flightCrtl.ser_land()
        
        print("OK.")
        rospy.spin()
    except IOError, KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
        btCm_n.saveData()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

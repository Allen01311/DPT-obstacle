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
    # title = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # allout = cv2.VideoWriter(title + '_all.avi', fourcc, 30,(856, 480))
    # bridge = CvBridge() 
    # strat = []
    path = [(24.9871405, 121.5734011)]
    #(24.9871405, 121.5734011) (200408 class image)
    #(24.9872064, 121.5735741) (toilet image)
    #path = [(24.986429, 121.572829), (24.986779, 121.572877), (24.987377, 121.572942)]
    end = [-1, -1]
    now_target = 0
    ini_height = 30.0
    # Take Off
    take0ff_complete = False
    finish = False
    go = 1
    # 0go 往返 回到起點降落
    # 1往 回到起點降落

    def __init__(self):
        self.tree = (
            self.NotFinish >> (self.isNotArrival|self.swichStage ) >> self.fixedPoseAndForward 
            | self.ChangeOrLanding
        )

        try:
          with open('btModel_BS.pkl', 'rb') as f: 
            model = pickle.load(f)
            self.loadData(model)
        except:
          print("First run")
        # self.title = '2020-02-04-DJI_rivertest'
        # self.fourcc = cv2.VideoWriter_fourcc('X',"V",'I','D')
        # self.out = cv2.VideoWriter(self.title + '_video.avi', self.fourcc, 30,(840, 490))
        # # sub
        # self.subscriber = rospy.Subscriber("/image/compressed", CompressedImage, self.image_cb,  queue_size = 1, buff_size=2**24)
        # self.pc2_sub = rospy.Subscriber("/orb_slam2_mono/map_points", PointCloud2, self.pc2_cb, queue_size = 1)

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


    # def image_cb(self, ros_data):
    #    np_arr = np.fromstring(ros_data.data, np.uint8)
    #    image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    #    #self.out.write(image_np)
    #    cv2.imshow('cv_img', image_np)
    #    cv2.waitKey(1)
        

    @condition
    def NotFinish(self):
        print("condition: NotFinish")
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
    
    @action
    def ChangeOrLanding(self):
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
        #dis = geopy.distance.vincenty(now, btControl_mission.drone.state.home).m
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
      
        print("arrival END 20m")
        
        t = Twist()
        btControl_mission.drone.flightCrtl.move_s(t, 0.5) 
        
        btControl_mission.finish = True
        
    @action
    def land(self):
        print("down to " + str(2.0))
        while btControl_mission.drone.state.alt > 2.5:
          tx = Twist()
          tx.linear.z = -1.0
          btControl_mission.drone.flightCrtl.move(tx,1)
        print("down to " + str(btControl_mission.drone.state.alt))
        btControl_mission.drone.flightCrtl.ser_land()
        
          
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
 
    def run(self):
        while True:
            if btControl_mission.finish == True:
                break
            bb = self.tree.blackboard(1)
            state = bb.tick()
            print("state = %s\n" % state)
            if btControl_mission.drone.isStop == True:
              exec("f = open(\"123.txt\",\'rb\')")           
            while state == RUNNING:
                state = bb.tick()
                print("state = %s\n" % state)
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
      
    btCm_n.up_to(20.0)
    #btCm_n.drone.flightCrtl.ser_land()    
    
    print("tune camera angle...")
    while btCm_n.drone.state.gimbal_pitch > -89:#-90.0: 
      btCm_n.drone.flightCrtl.ser_gimbal_90()  
    print("camera angle: ", btCm_n.drone.state.gimbal_pitch)
    
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
    except Exception as e:
        print("Shutting down ROS Image feature detector module")
        btCm_n.saveData()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

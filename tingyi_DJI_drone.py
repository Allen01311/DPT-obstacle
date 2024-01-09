#!/usr/bin/env python
import rospy

import cv2
import cv_bridge
import math
import time
import threading

from distutils.util import strtobool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Empty, Bool, Int8, Int32, String,Int16MultiArray, Float64MultiArray
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse

class DroneState:

    def __init__(self):
        self.isFlying = False
        self.isConnected = False
        self.areMotorsOn = False

        self.home = [500.0, 500.0]
        self.iniPose = -2.95  # indoor1: 1.93    indoor22: 0.35 
        self.battery = 100

        self.iniyaw = -1
        self.yaw = 0.0
        self.lat = 500.0
        self.lot = 500.0
        self.alt = 0.0
        self.indoor_height = 0.0
        self.new_height = 0.0 #231016 樓頂至無人機高度
        
        self.gimbal_pitch = 0.0  # up 29.2  ~  -90 down
        self.gimbal_yaw = 0.0
        self.gimbal_roll = 0.0
        
        self.index = 4
        self.target_type = -1
        self.tp_x = 160
        self.tp_y = 160
        self.grid_x = -1
        self.grid_y = -1
        
        self.rx = -1
        self.ry = -1
        self.cx = -1
        self.cy = -1
        self.px = -1
        self.py = -1
        self.followType = -1
        self.inContour = -1
        self.move_dis = 0
        
        self.found = 0
        self.tag_x = -1
        self.tag_y = -1
        self.tc_0 = -1
        self.tc_1 = -1
        self.tc_2 = -1
        self.tc_3 = -1
        self.tag_count = -1
        self.tag_type = -1
        self.tag_dis = -1
        self.tag_height = -1
        self.tag_direction = -1
        
        self.land_x = -1
        self.land_y = -1
        self.land_cx = -1
        self.land_cy = -1
        self.now_lap = -1
        self.laps = -1
        
        self.qrcode_mission = -1
        self.qrcode_direction = 0
        self.qrcode_lux = -1
        self.qrcode_luy = -1
        self.qrcode_ldx = -1
        self.qrcode_ldy = -1
        self.qrcode_cx = -1
        self.qrcode_cy = -1
        self.qrcode_0 = -1
        self.qrcode_1 = -1
        self.qrcode_2 = -1
        self.qrcode_3 = -1
        self.qrcode_4 = -1
        
        self.m2_red_x = -1
        self.m2_red_y = -1
        
        self.hl_tx = -1
        self.hl_ty = -1 
        self.hl_dis = -1
        self.hl_dis_x = -1
        self.hl_dis_y = -1
        self.hl_cy = -1
        self.hl_cx = -1
        
        self.hlHSV_tx = -1
        self.hlHSV_ty = -1 
        self.hlHSV_dis = -1
        self.hlHSV_dis_x = -1
        self.hlHSV_dis_y = -1
        #self.flightTimeInSec = 0

    def show(self):
      print(self.isFlying)
      print(self.isConnected)
      print(self.areMotorsOn)
      
      print(self.home)
      print(self.iniPose)
      print(self.battery)
      
      print(self.yaw)
      print(self.lat)
      print(self.lot)
      print(self.alt)
      print(self.indoor_height)
      print(self.new_height)
      
      print(self.gimbal_pitch)
      print(self.gimbal_yaw)
      print(self.gimbal_roll)

class FlightController:

    def move(self, twist, limitTime):
        limitTime = limitTime * 1000
        startTime = int(round(time.time()*1000))
        rate = rospy.Rate(10)
        # print "MOVE~~~~"
        pub_move_forward = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        while not rospy.is_shutdown():
            connections = pub_move_forward.get_num_connections()

            if connections > 0:
                endTime = int(round(time.time()*1000))
                if endTime - startTime < limitTime:
                    pub_move_forward.publish(twist)

                else:
                    twist.linear.x = 0
                    twist.linear.y = 0
                    twist.linear.z = 0
                    twist.angular.z = 0
                    #pub_move_forward.publish(twist)
                    break
            rate.sleep()
    
    def move_s(self, twist, limitTime):
        limitTime = limitTime * 1000
        startTime = int(round(time.time()*1000))
        rate = rospy.Rate(10)
        # print "MOVE~~~~"
        pub_move_forward = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        while not rospy.is_shutdown():
            connections = pub_move_forward.get_num_connections()
            
            if connections > 0:
                endTime = int(round(time.time()*1000))
                if endTime - startTime < limitTime:
                    pub_move_forward.publish(twist)

                else:
                    #twist.linear.x = 0
                    #twist.linear.y = 0
                    #twist.linear.z = 0
                    #twist.angular.z = 0
                    pub_move_forward.publish(Twist())
                    break
            rate.sleep()
    
    def lm_detection(self, b):
        rate = rospy.Rate(10)
        pub_controllm = rospy.Publisher("/control_landmark", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controllm.get_num_connections()

            if con > 0:
              pub_controllm.publish(Bool(b))
              rate.sleep()
              break
    
    def s_detection(self, b):
    
        rate = rospy.Rate(10)
        pub_controlC = rospy.Publisher("/control_color", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlC.get_num_connections()

            if con > 0:
              pub_controlC.publish(Bool(b))
              rate.sleep()
              break
              
    def land_detection(self, b):
    
        rate = rospy.Rate(10)
        pub_controlL = rospy.Publisher("/control_land", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlL.get_num_connections()

            if con > 0:
              pub_controlL.publish(Bool(b))
              rate.sleep()
              break
 
    def r_detection(self, b):
    
        rate = rospy.Rate(10)
        pub_controlR = rospy.Publisher("/control_road", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlR.get_num_connections()

            if con > 0:
              pub_controlR.publish(Bool(b))
              rate.sleep()
              break
              
    def red_detection(self, b):
    
        rate = rospy.Rate(10)
        pub_controlRed = rospy.Publisher("/control_red", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlRed.get_num_connections()

            if con > 0:
              pub_controlRed.publish(Bool(b))
              rate.sleep()
              break
              
    def update_mission(self, m):
        
        rate = rospy.Rate(10)
        pub_controlM = rospy.Publisher("/update_mission", Int8, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlM.get_num_connections()

            if con > 0:
              pub_controlM.publish(Int8(m))
              rate.sleep()
              break
    
    def controlSeg(self, b):
        rate = rospy.Rate(10)
        pub_controlSeg = rospy.Publisher("/controlSeg", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlSeg.get_num_connections()

            if con > 0:
              pub_controlSeg.publish(Bool(b))
              rate.sleep()
              break

    def start_collect_3(self, b):
        rate = rospy.Rate(10)
        pub_controlC3 = rospy.Publisher("/collect3", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlC3.get_num_connections()

            if con > 0:
              pub_controlC3.publish(Bool(b))
              rate.sleep()
              break
    
    def start_collect_4(self, b):
        rate = rospy.Rate(10)
        pub_controlC4 = rospy.Publisher("/collect4", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_controlC4.get_num_connections()

            if con > 0:
              pub_controlC4.publish(Bool(b))
              rate.sleep()
              break
    
    #231216
    def active_image(self, b):
        rate = rospy.Rate(10)
        pub_active_image = rospy.Publisher("/active_image", Bool, queue_size=10)
        while not rospy.is_shutdown():
            con = pub_active_image.get_num_connections()

            if con > 0:
              pub_active_image.publish(Bool(b))
              rate.sleep()
              break
                        
    def ser_gimbal_zero(self):
      rospy.wait_for_service('/flight_commands/gimbal_zero')
      try:
         call1 = rospy.ServiceProxy('/flight_commands/gimbal_zero', EmptySrv)
         resp1 = call1()
         return resp1
      except rospy.ServiceException as e:
         print("Service call failed: " + e)
         
    def ser_gimbal_90(self):
      rospy.wait_for_service('/flight_commands/gimbal_90')
      try:
         call1 = rospy.ServiceProxy('/flight_commands/gimbal_90', EmptySrv)
         resp1 = call1()
         return resp1
      except rospy.ServiceException as e:
         print("Service call failed: " + e)
         
    def ser_takeoff(self):
      rospy.wait_for_service('/flight_commands/takeoff')
      try:
         call1 = rospy.ServiceProxy('/flight_commands/takeoff', EmptySrv)
         resp1 = call1()
         return resp1
      except rospy.ServiceException as e:
         print("Service call failed: " + e)
         
    def ser_land(self):
      rospy.wait_for_service('/flight_commands/land')
      try:
         call1 = rospy.ServiceProxy('/flight_commands/land', EmptySrv)
         resp1 = call1()
         return resp1
      except rospy.ServiceException as e:
         print("Service call failed: " + e)
    
    def ser_stop(self):
      rospy.wait_for_service('/flight_commands/stop')
      try:
         call1 = rospy.ServiceProxy('/flight_commands/stop', EmptySrv)
         resp1 = call1()
         return resp1
      except rospy.ServiceException as e:
         print("Service call failed: " + e)

    def ser_start(self):
      rospy.wait_for_service('/flight_commands/start')
      try:
         call1 = rospy.ServiceProxy('/flight_commands/start', EmptySrv)
         resp1 = call1()
         return resp1
      except rospy.ServiceException as e:
         print("Service call failed: " + e)

    def active_river(self):
      rate = rospy.Rate(10)
      pub_active_river = rospy.Publisher("/active_river", Empty, queue_size=10)
      
      while not rospy.is_shutdown():
            con = pub_active_river.get_num_connections()

            if con > 0:
              pub_active_river.publish(Empty())
              rate.sleep()
              nt = rospy.get_time()
              while rospy.get_time() - nt < 0.1:
                pass
              break
            
class Drone():
    def __init__(self):

        self.isSetHome = False
        self.isSetYaw = False
        self.isStop = False
        self.iniSetCp = False
        self.count = 0
        self.receive_pose_time = 0.0
        self.state = DroneState()
        self.flightCrtl = FlightController()
        self._sensor()

    def _sensor(self):
        _status_sub = rospy.Subscriber("/dji/status", String, self._status_cb, queue_size = 5)
        _point_fira_tag = rospy.Subscriber("/target_point_fira_tag", Float64MultiArray, self._tp_fira_tag_cb, queue_size = 5)
        _point_fira_road = rospy.Subscriber("/target_point_fira_road", Float64MultiArray, self._tp_fira_road_cb ,queue_size = 5)
        _point_fira_land = rospy.Subscriber("/target_point_fira_land", Float64MultiArray, self._tp_fira_land_cb ,queue_size = 5)
        _test_sub = rospy.Subscriber("/test_signal", Int32, self._sig_cb, queue_size = 1)
        _qrcode_sub = rospy.Subscriber("/target_fira_2_qrcode", Float64MultiArray, self._qrcode_cb, queue_size = 1)
        _point_fira_red = rospy.Subscriber("/target_point_fira_red", Float64MultiArray, self._tp_fira_red_cb ,queue_size = 5)
        #_point_pub = rospy.Subscriber("/target_point", Int16MultiArray, self._tp_cb ,queue_size = 5)
        #_point_fira_pub = rospy.Subscriber("/target_point_fira", Float64MultiArray, self._tp_fira_cb ,queue_size = 1)
        _point_hl_sub = rospy.Subscriber("/target_hl", Float64MultiArray, self._tp_tl_cb ,queue_size = 5)
        _point_hlHSV_sub = rospy.Subscriber("/target_hlHSV", Float64MultiArray, self._HSVtp_tl_cb ,queue_size = 5)


    def _tp_tl_cb(self, msg):
      self.state.hl_tx = msg.data[0]
      self.state.hl_ty = msg.data[1]
      self.state.hl_dis = msg.data[2]
      self.state.hl_dis_x = msg.data[3]
      self.state.hl_dis_y = msg.data[4]
      self.state.hl_cy = msg.data[5]
      self.state.new_height = msg.data[6]
      
    def _HSVtp_tl_cb(self, msg):
      self.state.hlHSV_tx = msg.data[0]
      self.state.hlHSV_ty = msg.data[1]
      self.state.hlHSV_dis = msg.data[2]
      self.state.hlHSV_dis_x = msg.data[3]
      self.state.hlHSV_dis_y = msg.data[4]

    def _tp_fira_red_cb(self, msg):
      self.state.m2_red_x = msg.data[0]
      self.state.m2_red_y = msg.data[1]
    
    def _qrcode_cb(self, msg):
      self.state.qrcode_mission = msg.data[0]
      self.state.qrcode_direction = msg.data[1]
      self.state.qrcode_lux = msg.data[2]
      self.state.qrcode_luy = msg.data[3]
      self.state.qrcode_ldx = msg.data[4]
      self.state.qrcode_ldy = msg.data[5]
      self.state.qrcode_cx = msg.data[6]
      self.state.qrcode_cy = msg.data[7]
      self.state.qrcode_0 = msg.data[8]
      self.state.qrcode_1 = msg.data[9]
      self.state.qrcode_2 = msg.data[10]
      self.state.qrcode_3 = msg.data[11]
      self.state.qrcode_4 = msg.data[12]
    
    def _tp_cb(self, msg):
      self.state.index = msg.data[0]
      self.state.tp_x = msg.data[1]
      self.state.tp_y = msg.data[2]
      self.state.grid_x = msg.data[3]
      self.state.grid_y = msg.data[4]
    
    def _tp_fira_cb(self, msg):
      self.state.tp_x = msg.data[0]
      self.state.tp_y = msg.data[1]
      self.state.target_type = msg.data[2]
      self.state.index = msg.data[3]
      self.state.grid_x = msg.data[4]
      self.state.grid_y = msg.data[5]
    
    def _tp_fira_road_cb(self, msg):
      self.state.rx = msg.data[0]
      self.state.ry = msg.data[1]
      self.state.cx = msg.data[2]
      self.state.cy = msg.data[3]
      self.state.px = msg.data[4]
      self.state.py = msg.data[5]
      self.state.followType = msg.data[6]
      self.state.inContour = msg.data[7]
      self.state.move_dis = msg.data[8]
    
    def _tp_fira_tag_cb(self, msg):
      #print(msg)
      self.state.found = msg.data[0]
      self.state.tag_x = msg.data[1]
      self.state.tag_y = msg.data[2]   
      self.state.tc_lux = msg.data[3]
      self.state.tc_luy = msg.data[4]
      self.state.tc_ldx = msg.data[5]
      self.state.tc_ldy = msg.data[6]
      self.state.tag_count = msg.data[7]
      self.state.tag_type = msg.data[8]
      self.state.tag_dis = msg.data[9]
      self.state.tag_height = msg.data[10]
      self.state.tag_direction = msg.data[11] 
    
    def _tp_fira_land_cb(self, msg):
      self.state.land_x = msg.data[0]
      self.state.land_y = msg.data[1]
      self.state.land_cx = msg.data[2]
      self.state.land_cy = msg.data[3]
      self.state.now_lap = msg.data[4]
      self.state.laps = msg.data[5]
      
    def _sig_cb(self, msg):
      self.isStop = True

    def _status_cb(self, msg):
      temp_data = msg.data.split(";")
      #for i in range(0, len(temp_data)):
      self.state.home = [float(temp_data[0].split("=")[1]), float(temp_data[1].split("=")[1])] 
      if self.state.iniyaw == -1:
         self.state.iniyaw = float(temp_data[3].split("=")[1])
      self.state.yaw = float(temp_data[3].split("=")[1])
      self.state.battery = float(temp_data[4].split("=")[1])
      self.state.isConnected = strtobool(temp_data[5].split("=")[1])
      self.state.areMotorsOn = strtobool(temp_data[6].split("=")[1])
      self.state.isFlying = strtobool(temp_data[7].split("=")[1])
      self.state.lat = float(temp_data[8].split("=")[1])
      self.state.lot = float(temp_data[9].split("=")[1])
      self.state.alt = float(temp_data[10].split("=")[1])
      self.state.gimbal_pitch = float(temp_data[12].split("=")[1])
      self.state.gimbal_yaw = float(temp_data[13].split("=")[1])
      self.state.gimbal_roll = float(temp_data[11].split("=")[1])
      self.state.indoor_height = float(temp_data[14].split("=")[1])

    def takeoff(self):

        rate = rospy.Rate(10)
        pub_takeoff = rospy.Publisher("/bebop/takeoff", Empty, queue_size=1)

        while not rospy.is_shutdown():
            connections = pub_takeoff.get_num_connections()

            # rospy.loginfo('Connections: %d', connections)
            if connections > 0:
                pub_takeoff.publish(Empty())
                rospy.loginfo('TingYi, Takeing off')
                break
            rate.sleep()
            
    def correct(self, hl_dis, hl_tx, hl_ty, centerx, hl_cy, hl_dis_x, hl_dis_y, stag):
        #目標點:(hl_tx, hl_ty)
        #畫面中心點:(centerx, hl_cy)
        tx = Twist()
        print("ori", hl_tx, hl_ty, hl_dis_x, hl_dis_y, hl_dis)
        dis = hl_dis
        vx = 0.0
        vy = 0.0
        if (hl_tx - centerx) != 0:
          vx = (hl_tx - centerx) / abs(hl_tx - centerx)
        if (hl_ty - hl_cy) != 0:
          vy = -(hl_ty - hl_cy) / abs(hl_ty - hl_cy)
        
        max_xy = max(hl_dis_x, hl_dis_y)
        
        scale_factor = hl_dis_x
        
        #判斷X、Y哪邊比較遠
        if max_xy == hl_dis_x:
          scale_factor = hl_dis_x
        elif max_xy == hl_dis_y:
          scale_factor = hl_dis_y
        
        command_x = vy
        command_y = vx
        if scale_factor != 0:
          command_x = vy * hl_dis_y / scale_factor
          command_y = vx * hl_dis_x / scale_factor
        #command_x = vy * btControl_mission.drone.state.hl_dis_y / scale_factor
        #command_y = vx * btControl_mission.drone.state.hl_dis_x / scale_factor
        
        command_dis = math.sqrt(command_x**2 + command_y**2) 
        scale = dis / command_dis
        	
        # 階段倍率調整實驗 nc
        rate = 1
        if stag == "actionDown":
            rate = 0.5
        
        tx.linear.x = scale * command_x * 0.85 * rate
        tx.linear.y = scale * command_y * 0.85 * rate
        
        # 避免數值大於8會出現不動 保守起見設定7 0430
        if tx.linear.x > 7:
            tx.linear.x = 7
        elif tx.linear.x < -7:
            tx.linear.x = -7
            
        if tx.linear.y > 7:
            tx.linear.y = 7 
        elif tx.linear.y < -7:
            tx.linear.y = -7
               
        # tx.linear.z = -0.5  
        
        print("act, x: ", tx.linear.x, "  act, y: " , tx.linear.y)
        
        return tx

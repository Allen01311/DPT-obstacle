#!/usr/bin/env python
import roslib
import rospy

from geometry_msgs.msg import Twist
from pprint import pprint
from std_msgs.msg import Bool, Empty
from nav_msgs.msg import Odometry
from pynput.keyboard import Key, Listener
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse

rospy.init_node('DJI_telop', anonymous=True)
def on_press(key):
	print(key.char)
	if key.char== 'w':
		moveforward()
	if key.char== 't':
		takeoff()
	if key.char== 'l':
		land()
	if key.char== 's':
		moveback()
	if key.char== 'a':
		moveleft()
	if key.char== 'd':
		moveright()
	if key.char== 'q':
		moveup()
	if key.char== 'e':
		movedown()
	if key.char== 'z':
		headleft()
	if key.char== 'x':
		headright()
	if key.char == Key.space:
		stop1()
def camer_control():
	print("camer_control")
	rospy.wait_for_service('/flight_commands/gimbal_up')
	try:
		call1 = rospy.ServiceProxy('/flight_commands/gimbal_up', EmptySrv)
		resp1 = call1()
		return resp1
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)

def ser_gimbal_down(self):
	print("camer_control2")
	rospy.wait_for_service('/flight_commands/gimbal_down')
	try:
		call1 = rospy.ServiceProxy('/flight_commands/gimbal_down', EmptySrv)
		resp1 = call1()
		return resp1
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)

def on_release(key):
    #print(format(key))
    if key == Key.esc:
        # Stop listener
        return False

def land():
	print("land")
	rospy.wait_for_service('/flight_commands/land')
	try:
		call1 = rospy.ServiceProxy('/flight_commands/land', EmptySrv)
		resp1 = call1()
		return resp1
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)
def takeoff():
	print("take_off")
	rospy.wait_for_service('/flight_commands/takeoff')
	try:
		call1 = rospy.ServiceProxy('/flight_commands/takeoff', EmptySrv)
		resp1 = call1()
		return resp1
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)
def headleft():
    print("headleft")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.angular.z = -0.3
    pub.publish(vel_msg)
def headright():
    print("headright")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.angular.z = 0.3
    pub.publish(vel_msg)
def moveforward():  
    print("moveforward")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0.3
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    pub.publish(vel_msg)

def moveback():  
    print("movebackward")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = -0.3
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    pub.publish(vel_msg)
def moveleft():  
    print("moveleft")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0
    vel_msg.linear.y = -0.3
    vel_msg.linear.z = 0
    pub.publish(vel_msg)
def moveright():  
    print("moveright")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0.3
    vel_msg.linear.z = 0
    pub.publish(vel_msg)
def movedown():  
    print("movedown")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = -1.0
    pub.publish(vel_msg)
def moveup():
    print("moveup")
    vel_msg = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 1.0
    pub.publish(vel_msg)
def stop1():
	print("stop")
	rospy.wait_for_service('/flight_commands/stop')
	try:
		call1 = rospy.ServiceProxy('/flight_commands/stop', EmptySrv)
		resp1 = call1()
		return resp1
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)
   
if __name__ == '__main__':  
	with Listener(on_press=on_press,on_release=on_release) as listener:
    		listener.join()	

	print("dww")
	

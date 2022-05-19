import rospy
from geometry_msgs.msg import Twist
import time
rospy.init_node('robot_vel')
robot1_pub=rospy.publisher('robot1/cmd_vel',Twist,queue_size=5)
robot2_pub=rospy.publisher('robot2/cmd_vel',Twist,queue_size=5)
robot3_pub=rospy.publisher('robot3/cmd_vel',Twist,queue_size=5)
robot4_pub=rospy.publisher('robot4/cmd_vel',Twist,queue_size=5)
while True:
    vel1_msg=Twist()
    vel1_msg.linear.x=0
    vel1_msg.angular.z=0
    robot1_pub.publish(vel1_msg)

    vel2_msg=Twist()
    vel2_msg.linear.x=0
    vel2_msg.angular.z=0
    robot2_pub.publish(vel1_msg)

    vel3_msg=Twist()
    vel3_msg.linear.x=0
    vel3_msg.angular.z=0
    robot3_pub.publish(vel3_msg)

    vel4_msg=Twist()
    vel4_msg.linear.x=0
    vel4_msg.angular.z=0
    robot4_pub.publish(vel4_msg)

    time.sleep(1)

    vel1_msg=Twist()
    vel1_msg.linear.x=0.2
    vel1_msg.angular.z=0.1
    robot1_pub.publish(vel1_msg)

    vel2_msg=Twist()
    vel2_msg.linear.x=0.2
    vel2_msg.angular.z=0.1
    robot2_pub.publish(vel1_msg)

    vel3_msg=Twist()
    vel3_msg.linear.x=0.2
    vel3_msg.angular.z=0.1
    robot3_pub.publish(vel3_msg)

    vel4_msg=Twist()
    vel4_msg.linear.x=0.2
    vel4_msg.angular.z=0.1
    robot4_pub.publish(vel4_msg)

    time.sleep(1)
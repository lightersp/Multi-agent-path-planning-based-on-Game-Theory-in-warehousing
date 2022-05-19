#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from  nav_msgs.msg import Odometry
import time
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as R


rospy.init_node('robot_vel',anonymous=True)
agv1_pub = rospy.Publisher('/agv1/cmd_vel', Twist, queue_size=1)
agv2_pub = rospy.Publisher('/agv2/cmd_vel', Twist, queue_size=1)
agv3_pub = rospy.Publisher('/agv3/cmd_vel', Twist, queue_size=1)
agv4_pub = rospy.Publisher('/agv4/cmd_vel', Twist, queue_size=1)






obstacle=np.array([[5,0],[-5,0],[0,-5],[0,5]])
#goal=[1,1,0,1,-1,0,-1,1,0,-1,-1,0]
#goal=[-3,-3,0,-3,3,0,3,-3,0,3,3,0]     #3
goal=[8,-3,0,8,3,0,8,-2,0,8,2,0]        #1
# svo=[np.pi/3,np.pi/3,np.pi/3,np.pi/3]
svo=[np.pi/3,3*np.pi/8,np.pi/3,3*np.pi/8]

def nash_learn(state):
    T=1                                 #采样时间
    N=8                                 #MPC预测的时域=T*N
    n_control=2                         #每个智能体包含的两个控制输入v,omega
    n_state=3                           #每个智能体状态信息包括x,y,theta
    n_player=4                          #仿真四个AGV
    v_max=0.3                           #最大线速度
    omega_max=np.pi/4                   #最大角速度




    position = ca.SX.sym('po', n_state * n_player)   #代表4个智能体的状态
    action = ca.SX.sym('act', n_control * n_player)  #代表对于智能体的控制作用,线速度、角速度
    rhs=ca.vertcat(action[0]*np.cos(position[2]),action[0]*np.sin(position[2]),-action[1],
                   action[2]*np.cos(position[5]),action[2]*np.sin(position[5]),-action[3],
                   action[4]*np.cos(position[8]),action[4]*np.sin(position[8]),-action[5],
                   action[6]*np.cos(position[11]),action[6]*np.sin(position[11]),-action[7])
    f=ca.Function('f',[position,action],[rhs],['input_position','input_action'],['rhs'])

    x = ca.SX.sym('x', n_state * n_player, N + 1)
    x[:, 0] = state # 初始状态
    u = ca.SX.sym('y', n_control * n_player, N)
    #状态转移模型
    for i in range(N):
        f_value=f(x[:,i],u[:,i])
        x[:,i+1]=x[:,i]+T*f_value

    #定义奖惩函数
    ##各智能体之间的距离
    d_2_a1 = ca.Function('da1', [position], [ca.norm_2(position[0:2] - position[3:5]) +
                                                 ca.norm_2(position[0:2] - position[6:8]) +
                                                 ca.norm_2(position[0:2] - position[9:11])], ['input11'], ['d11'])
    d_2_a2 = ca.Function('da2', [position], [ca.norm_2(position[3:5] - position[0:2]) +
                                                 ca.norm_2(position[3:5] - position[6:8]) +
                                                 ca.norm_2(position[3:5] - position[9:11])], ['input12'], ['d12'])
    d_2_a3 = ca.Function('da3', [position], [ca.norm_2(position[6:8] - position[0:2]) +
                                                 ca.norm_2(position[6:8] - position[3:5]) +
                                                 ca.norm_2(position[6:8] - position[9:11])], ['input13'], ['d13'])
    d_2_a4 = ca.Function('da4', [position], [ca.norm_2(position[9:11] - position[0:2]) +
                                                 ca.norm_2(position[9:11] - position[3:5]) +
                                                 ca.norm_2(position[9:11] - position[6:8])], ['input14'], ['d14'])

    ##到各自目的地的距离
    d_2_goal1 = ca.Function('d1', [position], [ca.norm_2(position[0:2] - goal[0:2])], ['input1'], ['d1'])
    d_2_goal2 = ca.Function('d2', [position], [ca.norm_2(position[3:5] - goal[3:5])], ['input2'], ['d2'])
    d_2_goal3 = ca.Function('d1', [position], [ca.norm_2(position[6:8] - goal[6:8])], ['input3'], ['d3'])
    d_2_goal4 = ca.Function('d1', [position], [ca.norm_2(position[9:11] - goal[9:11])], ['input4'], ['d4'])

    ##总的时刻奖励
    # reward1 = ca.Function('r1', [position], [d_2_goal1(position) - 0.01 * d_2_a1(position)], ['agent_1_input'],
    #                       ['agent_1_r'])
    # reward2 = ca.Function('r2', [position], [d_2_goal2(position) - 0.01 * d_2_a2(position)], ['agent_2_input'],
    #                       ['agent_2_r'])
    # reward3 = ca.Function('r3', [position], [d_2_goal3(position) - 0.01 * d_2_a3(position)], ['agent_3_input'],
    #                       ['agent_3_r'])
    # reward4 = ca.Function('r4', [position], [d_2_goal4(position) - 0.01 * d_2_a4(position)], ['agent_4_input'],
    #                        ['agent_4_r'])
    reward1 = ca.Function('r1', [position], [d_2_goal1(position) ], ['agent_1_input'],
                          ['agent_1_r'])
    reward2 = ca.Function('r2', [position], [d_2_goal2(position) ], ['agent_2_input'],
                          ['agent_2_r'])
    reward3 = ca.Function('r3', [position], [d_2_goal3(position) ], ['agent_3_input'],
                          ['agent_3_r'])
    reward4 = ca.Function('r4', [position], [d_2_goal4(position) ], ['agent_4_input'],
                          ['agent_4_r'])


    ##时刻效用函数
    u1 = ca.Function('u1', [position], [(np.cos(svo[0]) * reward1(position) +
                                         np.sin(svo[0]) * reward2(position) +
                                         np.sin(svo[0]) * reward3(position) +
                                         np.sin(svo[0]) * reward4(position)) / 3], ['utility1_input'], ['utility1_output'])
    u2 = ca.Function('u2', [position], [(np.cos(svo[1]) * reward2(position) +
                                         np.sin(svo[1]) * reward1(position) +
                                         np.sin(svo[1]) * reward3(position) +
                                         np.sin(svo[1]) * reward4(position)) / 3], ['utility2_input'], ['utility2_output'])
    u3 = ca.Function('u3', [position], [(np.cos(svo[2]) * reward3(position) +
                                         np.sin(svo[2]) * reward2(position) +
                                         np.sin(svo[2]) * reward1(position) +
                                         np.sin(svo[2]) * reward4(position)) / 3], ['utility3_input'], ['utility3_output'])
    u4 = ca.Function('u4', [position], [(np.cos(svo[3]) * reward4(position) +
                                         np.sin(svo[3]) * reward2(position) +
                                         np.sin(svo[3]) * reward3(position) +
                                         np.sin(svo[3]) * reward1(position)) / 3], ['utility4_input'], ['utility4_output'])
    # 总效用
    u1_total = 0
    u2_total = 0
    u3_total = 0
    u4_total = 0
    for i in range(1, N+1):
        u1_total += +u1(x[:, i])
    # 智能体1整个时域上的效用函数
    for i in range(1, N+1):
        u2_total += +u2(x[:, i])
    # 智能体2整个时域上的效用函数
    for i in range(1, N+1):
        u3_total += +u3(x[:, i])
    # 智能体3整个时域上的效用函数
    for i in range(1, N+1):
        u4_total += +u4(x[:, i])
    # 智能体4整个时域上的效用函数
    obj = u1_total + u2_total + u3_total + u4_total
    # 目标函数

    #约束条件
    g = []
    ubg = []
    lbg = []
    ##地图边界限制
    for inter1 in range(1,N+1):
        g.append(x[0,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[1,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[3,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[4,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[6,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[7,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[9,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[10,inter1])
        ubg.append(9)
        lbg.append(-9)
    ##agv之间的距离
        g.append(ca.norm_2(x[0:2,inter1]-x[3:5,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[0:2,inter1]-x[6:8,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[0:2,inter1]-x[9:11,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[3:5,inter1]-x[6:8,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[3:5,inter1]-x[9:11,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[6:8,inter1]-x[9:11,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)
    ##与障碍物的距离
        for inter2 in range(4):
            g.append(ca.norm_2(x[0:2,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)
            g.append(ca.norm_2(x[3:5,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)
            g.append(ca.norm_2(x[6:8,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)
            g.append(ca.norm_2(x[9:11,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)


    lbx=[]
    ubx=[]

    for inter4 in range(n_player*N):
        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-omega_max)
        ubx.append(omega_max)
    #创建求解器
    nlp = {'f': obj, 'x': ca.reshape(u, -1, 1),'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp,opts_setting)
    r = solver(x0=np.zeros((n_control*n_player*N,1)), lbx=lbx, ubx=ubx, ubg=ubg, lbg=lbg)
    best_act = r['x']
    act_next = best_act[0:8]
    return act_next
def nash_learn_(state):
    T=1                                 #采样时间
    N=1                                 #MPC预测的时域=T*N
    n_control=2                         #每个智能体包含的两个控制输入v,omega
    n_state=3                           #每个智能体状态信息包括x,y,theta
    n_player=4                          #仿真四个AGV
    v_max=0.3                           #最大线速度
    omega_max=np.pi/4                   #最大角速度




    position = ca.SX.sym('po', n_state * n_player)   #代表4个智能体的状态
    action = ca.SX.sym('act', n_control * n_player)  #代表对于智能体的控制作用,线速度、角速度
    rhs=ca.vertcat(action[0]*np.cos(position[2]),action[0]*np.sin(position[2]),-action[1],
                   action[2]*np.cos(position[5]),action[2]*np.sin(position[5]),-action[3],
                   action[4]*np.cos(position[8]),action[4]*np.sin(position[8]),-action[5],
                   action[6]*np.cos(position[11]),action[6]*np.sin(position[11]),-action[7])
    f=ca.Function('f',[position,action],[rhs],['input_position','input_action'],['rhs'])

    x = ca.SX.sym('x', n_state * n_player, N + 1)
    x[:, 0] = state # 初始状态
    u = ca.SX.sym('y', n_control * n_player, N)
    #状态转移模型
    for i in range(N):
        f_value=f(x[:,i],u[:,i])
        x[:,i+1]=x[:,i]+T*f_value

    #定义奖惩函数
    ##各智能体之间的距离
    d_2_a1 = ca.Function('da1', [position], [ca.norm_2(position[0:2] - position[3:5]) +
                                                 ca.norm_2(position[0:2] - position[6:8]) +
                                                 ca.norm_2(position[0:2] - position[9:11])], ['input11'], ['d11'])
    d_2_a2 = ca.Function('da2', [position], [ca.norm_2(position[3:5] - position[0:2]) +
                                                 ca.norm_2(position[3:5] - position[6:8]) +
                                                 ca.norm_2(position[3:5] - position[9:11])], ['input12'], ['d12'])
    d_2_a3 = ca.Function('da3', [position], [ca.norm_2(position[6:8] - position[0:2]) +
                                                 ca.norm_2(position[6:8] - position[3:5]) +
                                                 ca.norm_2(position[6:8] - position[9:11])], ['input13'], ['d13'])
    d_2_a4 = ca.Function('da4', [position], [ca.norm_2(position[9:11] - position[0:2]) +
                                                 ca.norm_2(position[9:11] - position[3:5]) +
                                                 ca.norm_2(position[9:11] - position[6:8])], ['input14'], ['d14'])

    ##到各自目的地的距离
    d_2_goal1 = ca.Function('d1', [position], [ca.norm_2(position[0:2] - goal[0:2])], ['input1'], ['d1'])
    d_2_goal2 = ca.Function('d2', [position], [ca.norm_2(position[3:5] - goal[3:5])], ['input2'], ['d2'])
    d_2_goal3 = ca.Function('d1', [position], [ca.norm_2(position[6:8] - goal[6:8])], ['input3'], ['d3'])
    d_2_goal4 = ca.Function('d1', [position], [ca.norm_2(position[9:11] - goal[9:11])], ['input4'], ['d4'])

    ##总的时刻奖励
    # reward1 = ca.Function('r1', [position], [d_2_goal1(position) - 0.01 * d_2_a1(position)], ['agent_1_input'],
    #                       ['agent_1_r'])
    # reward2 = ca.Function('r2', [position], [d_2_goal2(position) - 0.01 * d_2_a2(position)], ['agent_2_input'],
    #                       ['agent_2_r'])
    # reward3 = ca.Function('r3', [position], [d_2_goal3(position) - 0.01 * d_2_a3(position)], ['agent_3_input'],
    #                       ['agent_3_r'])
    # reward4 = ca.Function('r4', [position], [d_2_goal4(position) - 0.01 * d_2_a4(position)], ['agent_4_input'],
    #                        ['agent_4_r'])
    reward1 = ca.Function('r1', [position], [d_2_goal1(position) ], ['agent_1_input'],
                          ['agent_1_r'])
    reward2 = ca.Function('r2', [position], [d_2_goal2(position) ], ['agent_2_input'],
                          ['agent_2_r'])
    reward3 = ca.Function('r3', [position], [d_2_goal3(position) ], ['agent_3_input'],
                          ['agent_3_r'])
    reward4 = ca.Function('r4', [position], [d_2_goal4(position) ], ['agent_4_input'],
                          ['agent_4_r'])


    ##时刻效用函数
    u1 = ca.Function('u1', [position], [(np.cos(svo[0]) * reward1(position) +
                                         np.sin(svo[0]) * reward2(position) +
                                         np.sin(svo[0]) * reward3(position) +
                                         np.sin(svo[0]) * reward4(position)) / 3], ['utility1_input'], ['utility1_output'])
    u2 = ca.Function('u2', [position], [(np.cos(svo[1]) * reward2(position) +
                                         np.sin(svo[1]) * reward1(position) +
                                         np.sin(svo[1]) * reward3(position) +
                                         np.sin(svo[1]) * reward4(position)) / 3], ['utility2_input'], ['utility2_output'])
    u3 = ca.Function('u3', [position], [(np.cos(svo[2]) * reward3(position) +
                                         np.sin(svo[2]) * reward2(position) +
                                         np.sin(svo[2]) * reward1(position) +
                                         np.sin(svo[2]) * reward4(position)) / 3], ['utility3_input'], ['utility3_output'])
    u4 = ca.Function('u4', [position], [(np.cos(svo[3]) * reward4(position) +
                                         np.sin(svo[3]) * reward2(position) +
                                         np.sin(svo[3]) * reward3(position) +
                                         np.sin(svo[3]) * reward1(position)) / 3], ['utility4_input'], ['utility4_output'])
    # 总效用
    u1_total = 0
    u2_total = 0
    u3_total = 0
    u4_total = 0
    for i in range(1, N+1):
        u1_total += +u1(x[:, i])
    # 智能体1整个时域上的效用函数
    for i in range(1, N+1):
        u2_total += +u2(x[:, i])
    # 智能体2整个时域上的效用函数
    for i in range(1, N+1):
        u3_total += +u3(x[:, i])
    # 智能体3整个时域上的效用函数
    for i in range(1, N+1):
        u4_total += +u4(x[:, i])
    # 智能体4整个时域上的效用函数
    obj = u1_total + u2_total + u3_total + u4_total
    # 目标函数

    #约束条件
    g = []
    ubg = []
    lbg = []
    ##地图边界限制
    for inter1 in range(1,N+1):
        g.append(x[0,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[1,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[3,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[4,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[6,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[7,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[9,inter1])
        ubg.append(9)
        lbg.append(-9)
        g.append(x[10,inter1])
        ubg.append(9)
        lbg.append(-9)
    ##agv之间的距离
        g.append(ca.norm_2(x[0:2,inter1]-x[3:5,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[0:2,inter1]-x[6:8,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[0:2,inter1]-x[9:11,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[3:5,inter1]-x[6:8,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[3:5,inter1]-x[9:11,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)

        g.append(ca.norm_2(x[6:8,inter1]-x[9:11,inter1]))
        lbg.append(0.5)
        ubg.append(np.inf)
    ##与障碍物的距离
        for inter2 in range(4):
            g.append(ca.norm_2(x[0:2,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)
            g.append(ca.norm_2(x[3:5,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)
            g.append(ca.norm_2(x[6:8,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)
            g.append(ca.norm_2(x[9:11,inter1]-obstacle[inter2]))
            lbg.append(1.65)
            ubg.append(np.inf)


    lbx=[]
    ubx=[]

    for inter4 in range(n_player*N):
        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-omega_max)
        ubx.append(omega_max)
    #创建求解器
    nlp = {'f': obj, 'x': ca.reshape(u, -1, 1),'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp,opts_setting)
    r = solver(x0=np.zeros((n_control*n_player*N,1)), lbx=lbx, ubx=ubx, ubg=ubg, lbg=lbg)
    best_act = r['x']
    act_next = best_act[0:8]
    return act_next







if __name__=='__main__':
    # for i in range(13):
    #     msg1=rospy.wait_for_message('/agv1/odom',Odometry,timeout=None)
    #     eu1 = R.from_quat([msg1.pose.pose.orientation.x, msg1.pose.pose.orientation.y, msg1.pose.pose.orientation.z,
    #                        msg1.pose.pose.orientation.w]).as_euler('zyx')
    #     pose1=np.array([msg1.pose.pose.position.x,msg1.pose.pose.position.y,eu1[0]])
    #     msg2=rospy.wait_for_message('/agv2/odom',Odometry,timeout=None)
    #     eu2 = R.from_quat([msg2.pose.pose.orientation.x, msg2.pose.pose.orientation.y, msg2.pose.pose.orientation.z,
    #                        msg2.pose.pose.orientation.w]).as_euler('zyx')
    #     pose2=np.array([msg2.pose.pose.position.x,msg2.pose.pose.position.y,eu2[0]])
    #     msg3=rospy.wait_for_message('/agv3/odom',Odometry,timeout=None)
    #     eu3 = R.from_quat([msg3.pose.pose.orientation.x, msg3.pose.pose.orientation.y, msg3.pose.pose.orientation.z,
    #                        msg3.pose.pose.orientation.w]).as_euler('zyx')
    #     pose3=np.array([msg3.pose.pose.position.x,msg3.pose.pose.position.y,eu3[0]])
    #     msg4=rospy.wait_for_message('/agv4/odom',Odometry,timeout=None)
    #     eu4 = R.from_quat([msg4.pose.pose.orientation.x, msg4.pose.pose.orientation.y, msg4.pose.pose.orientation.z,
    #                        msg4.pose.pose.orientation.w]).as_euler('zyx')
    #     pose4=np.array([msg4.pose.pose.position.x,msg4.pose.pose.position.y,eu4[0]])
    #
    #     state=np.array([*pose1,*pose2,*pose3,*pose4])
    #     rospy.loginfo(state)
    #     nash_equlibrium=nash_learn_(state)
    #
    #
    #     vel1_msg = Twist()
    #     vel1_msg.linear.x = nash_equlibrium[0]
    #     vel1_msg.angular.z=nash_equlibrium[1]
    #     agv1_pub.publish(vel1_msg)
    #
    #     vel2_msg = Twist()
    #     vel2_msg.linear.x = nash_equlibrium[2]
    #     vel2_msg.angular.z = nash_equlibrium[3]
    #     agv2_pub.publish(vel2_msg)
    #
    #     vel3_msg = Twist()
    #     vel3_msg.linear.x = nash_equlibrium[4]
    #     vel3_msg.angular.z = nash_equlibrium[5]
    #     agv3_pub.publish(vel3_msg)
    #
    #     vel4_msg = Twist()
    #     vel4_msg.linear.x = nash_equlibrium[6]
    #     vel4_msg.angular.z = nash_equlibrium[7]
    #     agv4_pub.publish(vel4_msg)
    #     #
    #     time.sleep(1)
    while not rospy.is_shutdown():


        msg1=rospy.wait_for_message('/agv1/odom',Odometry,timeout=None)
        eu1 = R.from_quat([msg1.pose.pose.orientation.x, msg1.pose.pose.orientation.y, msg1.pose.pose.orientation.z,
                           msg1.pose.pose.orientation.w]).as_euler('zyx')
        pose1=np.array([msg1.pose.pose.position.x,msg1.pose.pose.position.y,eu1[0]])
        msg2=rospy.wait_for_message('/agv2/odom',Odometry,timeout=None)
        eu2 = R.from_quat([msg2.pose.pose.orientation.x, msg2.pose.pose.orientation.y, msg2.pose.pose.orientation.z,
                           msg2.pose.pose.orientation.w]).as_euler('zyx')
        pose2=np.array([msg2.pose.pose.position.x,msg2.pose.pose.position.y,eu2[0]])
        msg3=rospy.wait_for_message('/agv3/odom',Odometry,timeout=None)
        eu3 = R.from_quat([msg3.pose.pose.orientation.x, msg3.pose.pose.orientation.y, msg3.pose.pose.orientation.z,
                           msg3.pose.pose.orientation.w]).as_euler('zyx')
        pose3=np.array([msg3.pose.pose.position.x,msg3.pose.pose.position.y,eu3[0]])
        msg4=rospy.wait_for_message('/agv4/odom',Odometry,timeout=None)
        eu4 = R.from_quat([msg4.pose.pose.orientation.x, msg4.pose.pose.orientation.y, msg4.pose.pose.orientation.z,
                           msg4.pose.pose.orientation.w]).as_euler('zyx')
        pose4=np.array([msg4.pose.pose.position.x,msg4.pose.pose.position.y,eu4[0]])

        state=np.array([*pose1,*pose2,*pose3,*pose4])
        rospy.loginfo(state)




        nash_equlibrium=nash_learn(state)


        vel1_msg = Twist()
        vel1_msg.linear.x = nash_equlibrium[0]
        vel1_msg.angular.z=nash_equlibrium[1]
        agv1_pub.publish(vel1_msg)

        vel2_msg = Twist()
        vel2_msg.linear.x = nash_equlibrium[2]
        vel2_msg.angular.z = nash_equlibrium[3]
        agv2_pub.publish(vel2_msg)

        vel3_msg = Twist()
        vel3_msg.linear.x = nash_equlibrium[4]
        vel3_msg.angular.z = nash_equlibrium[5]
        agv3_pub.publish(vel3_msg)

        vel4_msg = Twist()
        vel4_msg.linear.x = nash_equlibrium[6]
        vel4_msg.angular.z = nash_equlibrium[7]
        agv4_pub.publish(vel4_msg)
        #
        time.sleep(1)




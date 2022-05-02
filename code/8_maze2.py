# !/usr/bin/env python3
# -*- coding: utf-8 -*-
##环境
import numpy as np
import time
import sys
import casadi as ca
from torch.utils.tensorboard import SummaryWriter
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
writer=SummaryWriter('8_maze2')
UNIT = 20  # 各自的最小长与宽
MAZE_H = 21  # 格子的高
MAZE_W = 21  # 格子的宽

#起始点、目标点以及SVO
goal1=np.array([16 * UNIT, 18 * UNIT])
goal2=np.array([10 * UNIT, 18 * UNIT])
goal3=np.array([4 * UNIT, 18 * UNIT])
goal4=np.array([18 * UNIT, 18 * UNIT])
goal5=np.array([14 * UNIT, 18 * UNIT])
goal6=np.array([12 * UNIT, 18 * UNIT])
goal7=np.array([8 * UNIT, 18 * UNIT])
goal8=np.array([6 * UNIT, 18 * UNIT])
origin1 = np.array([70, 50])
origin2 = np.array([210, 50])
origin3 = np.array([350, 50])
origin4 = np.array([10,50])
origin5 = np.array([110,50])
origin6 = np.array([150,50])
origin7 = np.array([250,50])
origin8 = np.array([310,50])

svo=np.array([np.pi/8,np.pi/8,np.pi/8,np.pi/8,np.pi/8,np.pi/8,np.pi/8,np.pi/8])
#
# svo=np.array([np.pi/4,np.pi/4,np.pi/4,np.pi/4,np.pi/4,np.pi/4,np.pi/4,np.pi/4])

#svo=np.array([3*np.pi/8,3*np.pi/8,3*np.pi/8,3*np.pi/8,3*np.pi/8,3*np.pi/8,3*np.pi/8,3*np.pi/8])

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.title('Warehouse_8AGVs-2')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='oldlace',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
            points = [x1 + radius, y1,
                      x1 + radius, y1,
                      x2 - radius, y1,
                      x2 - radius, y1,
                      x2, y1,
                      x2, y1 + radius,
                      x2, y1 + radius,
                      x2, y2 - radius,
                      x2, y2 - radius,
                      x2, y2,
                      x2 - radius, y2,
                      x2 - radius, y2,
                      x1 + radius, y2,
                      x1 + radius, y2,
                      x1, y2,
                      x1, y2 - radius,
                      x1, y2 - radius,
                      x1, y1 + radius,
                      x1, y1 + radius,
                      x1, y1]

            return self.canvas.create_polygon(points, **kwargs, smooth=True, outline='black')
        #  构造地图
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        # 分拣站
        for i in range(1, 16, 7):
            self.canvas.create_rectangle(i * UNIT, 0, (i + 5) * UNIT, 2 * UNIT, fill='sandybrown')
        self.obstacle=[]
        # 障碍物货架
        for k in range(1, 17, 5):
            self.obstacle.append(np.array([k * UNIT, 15 * UNIT]))
            round_rectangle(k * UNIT, 15 * UNIT, (k + 2) * UNIT, 17 * UNIT, fill='seagreen',radius=70,width=1)
        self.obstacle.append(np.array([0 * UNIT, 4 * UNIT]))
        round_rectangle(0 * UNIT, 4 * UNIT, 9 * UNIT, 13 * UNIT, fill='seagreen', radius=120, width=2)
        self.obstacle.append(np.array([12 * UNIT, 4 * UNIT]))
        round_rectangle(12 * UNIT, 4 * UNIT, 21 * UNIT, 13 * UNIT, fill='seagreen', radius=120, width=2)
        self.canvas.create_line(0, 6 * UNIT, 8.9* UNIT, 6 * UNIT)
        self.canvas.create_line(0, 6.4 * UNIT, 8.9 * UNIT, 6.4 * UNIT)

        self.canvas.create_line(0, 10.6 * UNIT, 8.9 * UNIT, 10.6 * UNIT)
        self.canvas.create_line(0, 11 * UNIT, 8.9 * UNIT, 11 * UNIT)

        self.canvas.create_line(0, 6.4 * UNIT, 8.9 * UNIT, 10.6 * UNIT)
        self.canvas.create_line(8.9*UNIT, 6.4 * UNIT, 0, 10.6 * UNIT)

        self.canvas.create_line(12.1 * UNIT, 6 * UNIT, 21*UNIT, 6 * UNIT)
        self.canvas.create_line(12.1 * UNIT, 6.4 * UNIT, 21 * UNIT, 6.4 * UNIT)

        self.canvas.create_line(12.1 * UNIT, 10.6 * UNIT, 21 * UNIT, 10.6 * UNIT)
        self.canvas.create_line(12.1 * UNIT, 11 * UNIT, 21 * UNIT, 11 * UNIT)

        self.canvas.create_line(12.1 * UNIT, 6.4 * UNIT, 21 * UNIT, 10.6 * UNIT)
        self.canvas.create_line(21 * UNIT, 6.4 * UNIT, 12.1 * UNIT, 10.6 * UNIT)
        # 目的地
        self.target1 = self.canvas.create_rectangle(
            16 * UNIT, 18 * UNIT, 17 * UNIT, 19 * UNIT,
            fill='SkyBlue1')
        self.target2 = self.canvas.create_rectangle(
            10 * UNIT, 18 * UNIT, 11 * UNIT, 19 * UNIT,
            fill='SteelBlue2')
        self.target3 = self.canvas.create_rectangle(
            4 * UNIT, 18 * UNIT, 5 * UNIT, 19 * UNIT,
            fill='RoyalBlue1')
        self.target4 = self.canvas.create_rectangle(
            18 * UNIT, 18 * UNIT, 19 * UNIT, 19 * UNIT,
            fill='Midnightblue')
        self.target5 = self.canvas.create_rectangle(
            goal5[0], goal5[1], goal5[0]+20, goal5[1]+20,
            fill='yellow')
        self.target6 = self.canvas.create_rectangle(
            goal6[0], goal6[1], goal6[0] + 20, goal6[1] + 20,
            fill='gold')
        self.target7 = self.canvas.create_rectangle(
            goal7[0], goal7[1], goal7[0]+20, goal7[1]+20,
            fill='gold3')
        self.target8 = self.canvas.create_rectangle(
            goal8[0], goal8[1], goal8[0]+20, goal8[1]+20,
            fill='goldenrod')

        # 初始点
        self.org1 = self.canvas.create_rectangle(
            origin1[0] - 10, origin1[1] - 10,
            origin1[0] + 10, origin1[1] + 10)
        self.org2 = self.canvas.create_rectangle(
            origin2[0] - 10, origin2[1] - 10,
            origin2[0] + 10, origin2[1] + 10)
        self.org3 = self.canvas.create_rectangle(
            origin3[0] - 10, origin3[1] - 10,
            origin3[0] + 10, origin3[1] + 10)
        self.org4 = self.canvas.create_rectangle(
            origin4[0] - 10, origin4[1] - 10,
            origin4[0] + 10, origin4[1] + 10)
        self.org5 = self.canvas.create_rectangle(
            origin5[0] - 10, origin5[1] - 10,
            origin5[0] + 10, origin5[1] + 10)
        self.org6 = self.canvas.create_rectangle(
            origin6[0] - 10, origin6[1] - 10,
            origin6[0] + 10, origin6[1] + 10)
        self.org7 = self.canvas.create_rectangle(
            origin7[0] - 10, origin7[1] - 10,
            origin7[0] + 10, origin7[1] + 10)
        self.org8 = self.canvas.create_rectangle(
            origin8[0] - 10, origin8[1] - 10,
            origin8[0] + 10, origin8[1] + 10)

        #第1个AGV
        self.rect1 = self.canvas.create_rectangle(
            origin1[0] - 10, origin1[1] - 10,
            origin1[0] + 10, origin1[1] + 10,
            fill='SkyBlue1')

        # 第2个AGV
        self.rect2 = self.canvas.create_rectangle(
            origin2[0] - 10, origin2[1] - 10,
            origin2[0] + 10, origin2[1] + 10,
            fill='SteelBlue2')
        #第3个AGV
        self.rect3 = self.canvas.create_rectangle(
            origin3[0] - 10, origin3[1] - 10,
            origin3[0] + 10, origin3[1] + 10,
            fill='RoyalBlue1')
        #第4个AGV
        self.rect4 = self.canvas.create_rectangle(
            origin4[0] - 10, origin4[1] - 10,
            origin4[0] + 10, origin4[1] + 10,
            fill='Midnightblue')
        #第5个AGV
        self.rect5 = self.canvas.create_rectangle(
            origin5[0] - 10, origin5[1] - 10,
            origin5[0] + 10, origin5[1] + 10,
            fill='yellow')
        #第6个AGV
        self.rect6 = self.canvas.create_rectangle(
            origin6[0] - 10, origin6[1] - 10,
            origin6[0] + 10, origin6[1] + 10,
            fill='gold')
        #第7个AGV
        self.rect7 = self.canvas.create_rectangle(
            origin7[0] - 10, origin7[1] - 10,
            origin7[0] + 10, origin7[1] + 10,
            fill='gold3')
        #第8个AGV
        self.rect8 = self.canvas.create_rectangle(
            origin8[0] - 10, origin8[1] - 10,
            origin8[0] + 10, origin8[1] + 10,
            fill='goldenrod')
        # pack all
        self.canvas.pack()

    def resetRobot(self):
        def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):

            points = [x1 + radius, y1,
                      x1 + radius, y1,
                      x2 - radius, y1,
                      x2 - radius, y1,
                      x2, y1,
                      x2, y1 + radius,
                      x2, y1 + radius,
                      x2, y2 - radius,
                      x2, y2 - radius,
                      x2, y2,
                      x2 - radius, y2,
                      x2 - radius, y2,
                      x1 + radius, y2,
                      x1 + radius, y2,
                      x1, y2,
                      x1, y2 - radius,
                      x1, y2 - radius,
                      x1, y1 + radius,
                      x1, y1 + radius,
                      x1, y1]

            return self.canvas.create_polygon(points, **kwargs, smooth=True)
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)
        self.canvas.delete(self.rect3)
        self.canvas.delete(self.rect4)
        self.canvas.delete(self.rect5)
        self.canvas.delete(self.rect6)
        self.canvas.delete(self.rect7)
        self.canvas.delete(self.rect8)
        self.rect1 = round_rectangle(origin1[0] - 10, origin1[1] - 10,
                                   origin1[0] + 10, origin1[1] + 10,
                                   fill='SkyBlue1',radius=3,outline='black')
        self.rect2 = round_rectangle(origin2[0] - 10, origin2[1] - 10,
                                     origin2[0] + 10, origin2[1] + 10,
                                     fill='SteelBlue2', radius=3, outline='black')
        self.rect3 = round_rectangle(origin3[0] - 10, origin3[1] - 10,
                                     origin3[0] + 10, origin3[1] + 10,
                                     fill='RoyalBlue', radius=3, outline='black')
        self.rect4 = round_rectangle(origin4[0] - 10, origin4[1] - 10,
                                     origin4[0] + 10, origin4[1] + 10,
                                     fill='Midnightblue', radius=3, outline='black')
        self.rect5 = round_rectangle(origin5[0] - 10, origin5[1] - 10,
                                     origin5[0] + 10, origin5[1] + 10,
                                     fill='yellow', radius=3, outline='black')
        self.rect6 = round_rectangle(origin6[0] - 10, origin6[1] - 10,
                                     origin6[0] + 10, origin6[1] + 10,
                                     fill='gold', radius=3, outline='black')
        self.rect7 = round_rectangle(origin7[0] - 10, origin7[1] - 10,
                                     origin7[0] + 10, origin7[1] + 10,
                                     fill='gold3', radius=3, outline='black')
        self.rect8 = round_rectangle(origin8[0] - 10, origin8[1] - 10,
                                     origin8[0] + 10, origin8[1] + 10,
                                     fill='goldenrod', radius=3, outline='black')
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2), self.canvas.coords(self.rect3),\
               self.canvas.coords(self.rect4), self.canvas.coords(self.rect5), self.canvas.coords(self.rect6),\
        self.canvas.coords(self.rect7),self.canvas.coords(self.rect8)
    #可视化渲染
    def render(self):
        time.sleep(0.01)
        self.update()
def  nash_learn(state):#通过casadi找最优控制
    T = 1  # 系统采样时间
    N = 3  # 需要预测的控制步长
    n_player=8#智能体数目
    n_state=2#每个智能体的状态数目即x,y
    n_control=2#每个智能体的控制作用即v_x,v_y
    x=ca.SX.sym('x',n_state*n_player,N+1)
    x[:,0]=state  #初始状态
    u=ca.SX.sym('y',n_control*n_player,N)


    #定义运动学模型

    position = ca.SX.sym('po', n_state * n_player)#共8个代表4个智能体的状态
    action = ca.SX.sym('act', n_control * n_player)#共8个代表对于智能体的控制作用，即，速度
    rhs = position + action * T
    move=ca.Function('move', [position, action], [rhs], ['input_state', 'control_input'], ['rhs'])
    for i in range(N):
        x[:, i + 1] = move(x[:, i], u[:, i])


    #即时奖励值

    goal=np.array([*goal1,*goal2,*goal3,*goal4,*goal5,*goal6,*goal7,*goal8])
        #各智能体到目标点的距离
    d_2_goal1 = ca.Function('d1', [position], [ca.norm_2(position[0:2] - goal[0:2])], ['input1'], ['d1'])
    d_2_goal2 = ca.Function('d2', [position], [ca.norm_2(position[2:4] - goal[2:4])], ['input2'], ['d2'])
    d_2_goal3 = ca.Function('d3', [position], [ca.norm_2(position[4:6] - goal[4:6])], ['input3'], ['d3'])
    d_2_goal4 = ca.Function('d4', [position], [ca.norm_2(position[6:8] - goal[6:8])], ['input4'], ['d4'])
    d_2_goal5 = ca.Function('d5', [position], [ca.norm_2(position[8:10] - goal[8:10])], ['input5'], ['d5'])
    d_2_goal6 = ca.Function('d6', [position], [ca.norm_2(position[10:12] - goal[10:12])], ['input6'], ['d6'])
    d_2_goal7 = ca.Function('d7', [position], [ca.norm_2(position[12:14] - goal[12:14])], ['input7'], ['d7'])
    d_2_goal8 = ca.Function('d8', [position], [ca.norm_2(position[14:] - goal[14:])], ['input8'], ['d8'])
    # 各智能体之间的距离
    d_2_a1 = ca.Function('da1', [position], [ca.norm_2(position[0:2] - position[2:4]) +
                                             ca.norm_2(position[0:2] - position[4:6]) +
                                             ca.norm_2(position[0:2] - position[6:8]) +
                                             ca.norm_2(position[0:2] - position[8:10]) +
                                             ca.norm_2(position[0:2] - position[10:12]) +
                                             ca.norm_2(position[0:2] - position[12:14]) +
                                             ca.norm_2(position[0:2] - position[14:])], ['input11'], ['d11'])
    d_2_a2 = ca.Function('da2', [position], [ca.norm_2(position[2:4] - position[0:2]) +
                                             ca.norm_2(position[2:4] - position[4:6]) +
                                             ca.norm_2(position[2:4] - position[6:8]) +
                                             ca.norm_2(position[2:4] - position[8:10]) +
                                             ca.norm_2(position[2:4] - position[10:12]) +
                                             ca.norm_2(position[2:4] - position[12:14]) +
                                             ca.norm_2(position[2:4] - position[14:])], ['input12'], ['d12'])
    d_2_a3 = ca.Function('da3', [position], [ca.norm_2(position[4:6] - position[0:2]) +
                                             ca.norm_2(position[4:6] - position[2:4]) +
                                             ca.norm_2(position[4:6] - position[6:8]) +
                                             ca.norm_2(position[4:6] - position[8:10]) +
                                             ca.norm_2(position[4:6] - position[10:12]) +
                                             ca.norm_2(position[4:6] - position[12:14]) +
                                             ca.norm_2(position[4:6] - position[14:])], ['input13'], ['d13'])
    d_2_a4 = ca.Function('da4', [position], [ca.norm_2(position[6:8] - position[0:2]) +
                                             ca.norm_2(position[6:8] - position[2:4]) +
                                             ca.norm_2(position[6:8] - position[4:6]) +
                                             ca.norm_2(position[6:8] - position[8:10]) +
                                             ca.norm_2(position[6:8] - position[10:12]) +
                                             ca.norm_2(position[6:8] - position[12:14]) +
                                             ca.norm_2(position[6:8] - position[14:])], ['input14'], ['d14'])
    d_2_a5 = ca.Function('da5', [position], [ca.norm_2(position[8:10] - position[0:2]) +
                                             ca.norm_2(position[8:10] - position[2:4]) +
                                             ca.norm_2(position[8:10] - position[4:6]) +
                                             ca.norm_2(position[8:10] - position[6:8]) +
                                             ca.norm_2(position[8:10] - position[10:12]) +
                                             ca.norm_2(position[8:10] - position[12:14]) +
                                             ca.norm_2(position[8:10] - position[14:])], ['input15'], ['d15'])
    d_2_a6 = ca.Function('da6', [position], [ca.norm_2(position[10:12] - position[0:2]) +
                                             ca.norm_2(position[10:12] - position[2:4]) +
                                             ca.norm_2(position[10:12] - position[4:6]) +
                                             ca.norm_2(position[10:12] - position[6:8]) +
                                             ca.norm_2(position[10:12] - position[8:10]) +
                                             ca.norm_2(position[10:12] - position[12:14]) +
                                             ca.norm_2(position[10:12] - position[14:])], ['input16'], ['d16'])
    d_2_a7 = ca.Function('da7', [position], [ca.norm_2(position[12:14] - position[0:2]) +
                                             ca.norm_2(position[12:14] - position[2:4]) +
                                             ca.norm_2(position[12:14] - position[4:6]) +
                                             ca.norm_2(position[12:14] - position[6:8]) +
                                             ca.norm_2(position[12:14] - position[8:10]) +
                                             ca.norm_2(position[12:14] - position[10:12]) +
                                             ca.norm_2(position[12:14] - position[14:])], ['input17'], ['d17'])
    d_2_a8 = ca.Function('da8', [position], [ca.norm_2(position[14:] - position[0:2]) +
                                             ca.norm_2(position[14:] - position[2:4]) +
                                             ca.norm_2(position[14:] - position[4:6]) +
                                             ca.norm_2(position[14:] - position[6:8]) +
                                             ca.norm_2(position[14:] - position[8:10]) +
                                             ca.norm_2(position[14:] - position[10:12]) +
                                             ca.norm_2(position[14:] - position[12:14])], ['input18'], ['d18'])
        #总的奖励值
    # reward1 = ca.Function('r1', [position], [d_2_goal1(position) - 0.001*d_2_a1(position)], ['agent_1_input'], ['agent_1_r'])
    # reward2 = ca.Function('r2', [position], [d_2_goal2(position) - 0.001*d_2_a2(position)], ['agent_2_input'], ['agent_2_r'])
    # reward3 = ca.Function('r3', [position], [d_2_goal3(position) - 0.001*d_2_a3(position)], ['agent_3_input'], ['agent_3_r'])
    # reward4 = ca.Function('r4', [position], [d_2_goal4(position) - 0.001*d_2_a4(position)], ['agent_4_input'], ['agent_4_r'])
    # reward5 = ca.Function('r4', [position], [d_2_goal5(position) - 0.001*d_2_a5(position)], ['agent_5_input'], ['agent_5_r'])
    # reward6 = ca.Function('r4', [position], [d_2_goal6(position) - 0.001*d_2_a6(position)], ['agent_6_input'], ['agent_6_r'])
    # reward7 = ca.Function('r4', [position], [d_2_goal7(position) - 0.001*d_2_a7(position)], ['agent_7_input'], ['agent_7_r'])
    # reward8 = ca.Function('r4', [position], [d_2_goal8(position) - 0.001*d_2_a8(position)], ['agent_8_input'], ['agent_8_r'])
       #备用奖励函数以备不收敛的情况发生orz
    reward1 = ca.Function('r1', [position], [d_2_goal1(position)], ['agent_1_input'], ['agent_1_r'])
    reward2 = ca.Function('r2', [position], [d_2_goal2(position)], ['agent_2_input'], ['agent_2_r'])
    reward3 = ca.Function('r3', [position], [d_2_goal3(position) ], ['agent_3_input'], ['agent_3_r'])
    reward4 = ca.Function('r4', [position], [d_2_goal4(position) ], ['agent_4_input'], ['agent_4_r'])
    reward5 = ca.Function('r5', [position], [d_2_goal5(position) ], ['agent_5_input'], ['agent_5_r'])
    reward6 = ca.Function('r6', [position], [d_2_goal6(position) ], ['agent_6_input'], ['agent_6_r'])
    reward7 = ca.Function('r7', [position], [d_2_goal7(position) ], ['agent_7_input'], ['agent_7_r'])
    reward8 = ca.Function('r8', [position], [d_2_goal8(position) ], ['agent_8_input'], ['agent_8_r'])

    #即时效用函数

    u1=ca.Function('u1',[position],[(np.cos(svo[0])*reward1(position)+
                                    np.sin(svo[0])*reward2(position)+
                                    np.sin(svo[0])*reward3(position)+
                                    np.sin(svo[0])*reward4(position)+
                                     np.sin(svo[0])*reward5(position)+
                                     np.sin(svo[0])*reward6(position)+
                                     np.sin(svo[0])*reward7(position)+
                                     np.sin(svo[0])*reward8(position))/7],['utility1_input'],['utility1_output'])
    u2=ca.Function('u2',[position],[(np.cos(svo[1])*reward2(position)+
                                    np.sin(svo[1])*reward1(position)+
                                    np.sin(svo[1])*reward3(position)+
                                    np.sin(svo[1])*reward4(position)+
                                     np.sin(svo[1])*reward5(position)+
                                     np.sin(svo[1])*reward6(position)+
                                     np.sin(svo[1])*reward7(position)+
                                     np.sin(svo[1])*reward8(position))/7],['utility2_input'],['utility2_output'])
    u3=ca.Function('u3',[position],[(np.cos(svo[2])*reward3(position)+
                                    np.sin(svo[2])*reward2(position)+
                                    np.sin(svo[2])*reward1(position)+
                                    np.sin(svo[2])*reward4(position)+
                                     np.sin(svo[2])*reward5(position)+
                                     np.sin(svo[2])*reward6(position)+
                                     np.sin(svo[2])*reward7(position)+
                                     np.sin(svo[2])*reward8(position))/7],['utility3_input'],['utility3_output'])
    u4=ca.Function('u4',[position],[(np.cos(svo[3])*reward4(position)+
                                    np.sin(svo[3])*reward2(position)+
                                    np.sin(svo[3])*reward3(position)+
                                    np.sin(svo[3])*reward1(position)+
                                     np.sin(svo[3])*reward5(position)+
                                     np.sin(svo[3])*reward6(position)+
                                     np.sin(svo[3])*reward7(position)+
                                     np.sin(svo[3])*reward8(position))/7],['utility4_input'],['utility4_output'])
    u5=ca.Function('u5',[position],[(np.cos(svo[4])*reward5(position)+
                                    np.sin(svo[4])*reward1(position)+
                                    np.sin(svo[4])*reward2(position)+
                                    np.sin(svo[4])*reward3(position)+
                                     np.sin(svo[4])*reward4(position)+
                                     np.sin(svo[4])*reward6(position)+
                                     np.sin(svo[4])*reward7(position)+
                                     np.sin(svo[4])*reward8(position))/7],['utility5_input'],['utility5_output'])
    u6=ca.Function('u6',[position],[(np.cos(svo[5])*reward6(position)+
                                    np.sin(svo[5])*reward1(position)+
                                    np.sin(svo[5])*reward2(position)+
                                    np.sin(svo[5])*reward3(position)+
                                     np.sin(svo[5])*reward4(position)+
                                     np.sin(svo[5])*reward5(position)+
                                     np.sin(svo[5])*reward7(position)+
                                     np.sin(svo[5])*reward8(position))/7],['utility6_input'],['utility6_output'])
    u7=ca.Function('u7',[position],[(np.cos(svo[6])*reward7(position)+
                                    np.sin(svo[6])*reward1(position)+
                                    np.sin(svo[6])*reward2(position)+
                                    np.sin(svo[6])*reward3(position)+
                                     np.sin(svo[6])*reward4(position)+
                                     np.sin(svo[6])*reward5(position)+
                                     np.sin(svo[6])*reward6(position)+
                                     np.sin(svo[6])*reward8(position))/7],['utility7_input'],['utility7_output'])
    u8=ca.Function('u8',[position],[(np.cos(svo[7])*reward8(position)+
                                    np.sin(svo[7])*reward1(position)+
                                    np.sin(svo[7])*reward2(position)+
                                    np.sin(svo[7])*reward3(position)+
                                     np.sin(svo[7])*reward4(position)+
                                     np.sin(svo[7])*reward5(position)+
                                     np.sin(svo[7])*reward6(position)+
                                     np.sin(svo[7])*reward7(position))/7],['utility8_input'],['utility8_output'])
    #总效用
    u1_total = 0;u2_total=0;u3_total=0;u4_total=0;u5_total=0;u6_total=0;u7_total=0;u8_total=0
    for i in range(1, 4):
        u1_total += +u1(x[:, i])
    #智能体1整个时域上的效用函数
    for i in range(1, 4):
        u2_total += +u2(x[:, i])
    #智能体2整个时域上的效用函数
    for i in range(1, 4):
        u3_total += +u3(x[:, i])
    #智能体3整个时域上的效用函数
    for i in range(1, 4):
        u4_total += +u4(x[:, i])
    #智能体4整个时域上的效用函数
    for i in range(1, 4):
        u5_total += +u5(x[:, i])
    #智能体5整个时域上的效用函数
    for i in range(1, 4):
        u6_total += +u6(x[:, i])
    #智能体6整个时域上的效用函数
    for i in range(1, 4):
        u7_total += +u7(x[:, i])
    #智能体7整个时域上的效用函数
    for i in range(1, 4):
        u8_total += +u8(x[:, i])
    #智能体8整个时域上的效用函数
    obj=u1_total + u2_total + u3_total + u4_total + u5_total +u6_total +u7_total + u8_total
    #约束条件
     ##状态限制,不能出界
    g=[]
    ubg=[]
    lbg=[]
    #分别存储每个AGV的约束，便于后续KKT条件的编写

    for inter1 in range(n_player*n_state):
         for inter2 in range(N):
             g.append(x[inter1,inter2])
             ubg.append(20*20)
             lbg.append(0)
     ##智能体间的距离
    interval=np.array([10,10])#智能体间的中心偏移
    interval2=np.array(([20,20]))#智能体与货架间的中心偏移
    interval3 = np.array(([90, 90]))  # 智能体与货架间的中心偏移
    for inter3 in range(1,4):
        g.append(ca.norm_2((x[0:2,inter3] + interval) - (x[2:4,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[4:6,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[6:8,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[8:10,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[10:12,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[12:14,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)

        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[4:6,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[6:8,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[8:10,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[10:12,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[12:14,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)

        g.append(ca.norm_2((x[4:6, inter3] + interval) - (x[6:8,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[4:6, inter3] + interval) - (x[8:10,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[4:6, inter3] + interval) - (x[10:12,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[4:6, inter3] + interval) - (x[12:14,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[4:6, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)

        g.append(ca.norm_2((x[6:8, inter3] + interval) - (x[8:10,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[6:8, inter3] + interval) - (x[10:12,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[6:8, inter3] + interval) - (x[12:14,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[6:8, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)

        g.append(ca.norm_2((x[8:10, inter3] + interval) - (x[10:12,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[8:10, inter3] + interval) - (x[12:14,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[8:10, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)

        g.append(ca.norm_2((x[10:12, inter3] + interval) - (x[12:14,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[10:12, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)

        g.append(ca.norm_2((x[12:14, inter3] + interval) - (x[14:16,inter3] + interval))-26)
        lbg.append(0)
        ubg.append(np.inf)
        #
        # #与障碍物的距离
        for inter4 in range(4):
            g.append(ca.norm_2((x[0:2, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[2:4, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[4:6, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[6:8, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[8:10, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[10:12, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[12:14, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[14:16, inter3] + interval) - (env.obstacle[inter4] + interval2))-30)
            lbg.append(0)
            ubg.append(np.inf)
        for inter5 in range(4,6):
            g.append(ca.norm_2((x[0:2, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[2:4, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[4:6, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[6:8, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[8:10, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[10:12, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[12:14, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)
            g.append(ca.norm_2((x[14:16, inter3] + interval) - (env.obstacle[inter5] + interval3)) - 120)
            lbg.append(0)
            ubg.append(np.inf)

    lbx=-3
    ubx=3

    nlp = {'f': obj, 'x': ca.reshape(u, -1, 1),'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp,opts_setting)

    r = solver(x0=np.zeros((n_control*n_player,N)).reshape(-1,1),lbx=lbx,ubx=ubx,ubg=ubg,lbg=lbg)
    best_act=r['x']
    act_next=best_act[0:16]
    return act_next




def move(action):
    env.canvas.move(env.rect1, action[0],action[1])
    env.canvas.move(env.rect2, action[2],action[3])
    env.canvas.move(env.rect3, action[4],action[5])
    env.canvas.move(env.rect4, action[6],action[7])
    env.canvas.move(env.rect5, action[8],action[9])
    env.canvas.move(env.rect6, action[10],action[11])
    env.canvas.move(env.rect7, action[12],action[13])
    env.canvas.move(env.rect8, action[14],action[15])
    state = [*env.canvas.coords(env.rect1)[0:2],
             *env.canvas.coords(env.rect2)[0:2],
             *env.canvas.coords(env.rect3)[0:2],
             *env.canvas.coords(env.rect4)[0:2],
             *env.canvas.coords(env.rect5)[0:2],
             *env.canvas.coords(env.rect6)[0:2],
             *env.canvas.coords(env.rect7)[0:2],
             *env.canvas.coords(env.rect8)[0:2]]
    print('agent1{0}agent2{1}agent3{2}agent4{3}agent5{4}agent6{5}agent7{6}agent8{7}'.format(state[:2],
                                                                                                     state[2:4],
                                                                                                     state[4:6],
                                                                                                     state[6:8],
                                                                                                     state[8:10],
                                                                                                     state[10:12],
                                                                                                     state[12:14],
                                                                                                     state[14:16]))

    return state
def compute_obj(state):
    state=np.array(state)
    goal = np.array([*goal1, *goal2, *goal3, *goal4,*goal5,*goal6,*goal7,*goal8])
    # 各智能体到目标点的距离
    d_2_goal1 = np.linalg.norm(state[0:2] - goal[0:2])
    d_2_goal2 = np.linalg.norm(state[2:4] - goal[2:4])
    d_2_goal3 = np.linalg.norm(state[4:6] - goal[4:6])
    d_2_goal4 = np.linalg.norm(state[6:8] - goal[6:8])
    d_2_goal6 = np.linalg.norm(state[10:12] - goal[10:12])
    d_2_goal7 = np.linalg.norm(state[12:14] - goal[12:14])
    d_2_goal8 = np.linalg.norm(state[14:16] - goal[14:16])
    d_2_goal5 = np.linalg.norm(state[8:10] - goal[8:10])

    # 总的奖励值
    reward1 = d_2_goal1
    reward2 = d_2_goal2
    reward3 = d_2_goal3
    reward4 = d_2_goal4
    reward5 = d_2_goal5
    reward6 = d_2_goal6
    reward7 = d_2_goal7
    reward8 = d_2_goal8


    u1 = (np.cos(svo[0])*reward1+np.sin(svo[0])*reward2+np.sin(svo[0])*reward3+np.sin(svo[0])*reward4+np.sin(svo[0])*reward5+np.sin(svo[0])*reward6+
                                     np.sin(svo[0])*reward7+
                                     np.sin(svo[0])*reward8)/7
    u2 = (np.cos(svo[1])*reward2+
                                    np.sin(svo[1])*reward1+
                                    np.sin(svo[1])*reward3+
                                    np.sin(svo[1])*reward4+
                                     np.sin(svo[1])*reward5+
                                     np.sin(svo[1])*reward6+
                                     np.sin(svo[1])*reward7+
                                     np.sin(svo[1])*reward8)/7
    u3 = (np.cos(svo[2])*reward3+
                                    np.sin(svo[2])*reward2+
                                    np.sin(svo[2])*reward1+
                                    np.sin(svo[2])*reward4+
                                     np.sin(svo[2])*reward5+
                                     np.sin(svo[2])*reward6+
                                     np.sin(svo[2])*reward7+
                                     np.sin(svo[2])*reward8)/7
    u4 = (np.cos(svo[3])*reward4+
                                    np.sin(svo[3])*reward2+
                                    np.sin(svo[3])*reward3+
                                    np.sin(svo[3])*reward1+
                                     np.sin(svo[3])*reward5+
                                     np.sin(svo[3])*reward6+
                                     np.sin(svo[3])*reward7+
                                     np.sin(svo[3])*reward8)/7
    u5=(np.cos(svo[4])*reward5+
                                    np.sin(svo[4])*reward1+
                                    np.sin(svo[4])*reward2+
                                    np.sin(svo[4])*reward3+
                                     np.sin(svo[4])*reward4+
                                     np.sin(svo[4])*reward6+
                                     np.sin(svo[4])*reward7+
                                     np.sin(svo[4])*reward8)/7
    u6=(np.cos(svo[5])*reward6+
                                    np.sin(svo[5])*reward1+
                                    np.sin(svo[5])*reward2+
                                    np.sin(svo[5])*reward3+
                                     np.sin(svo[5])*reward4+
                                     np.sin(svo[5])*reward5+
                                     np.sin(svo[5])*reward7+
                                     np.sin(svo[5])*reward8)/7
    u7=(np.cos(svo[6])*reward7+
                                    np.sin(svo[6])*reward1+
                                    np.sin(svo[6])*reward2+
                                    np.sin(svo[6])*reward3+
                                     np.sin(svo[6])*reward4+
                                     np.sin(svo[6])*reward5+
                                     np.sin(svo[6])*reward6+
                                     np.sin(svo[6])*reward8)/7
    u8=(np.cos(svo[7])*reward8+
                                    np.sin(svo[7])*reward1+
                                    np.sin(svo[7])*reward2+
                                    np.sin(svo[7])*reward3+
                                     np.sin(svo[7])*reward4+
                                     np.sin(svo[7])*reward5+
                                     np.sin(svo[7])*reward6+
                                     np.sin(svo[7])*reward7)/7
    obj=-u1 - u2 - u3 - u4-u5-u6-u7
    return obj,-u1,-u2,-u3,-u4,-u5,-u6,-u7,-u8


env = Maze()
def find_way():
    #复位
    env.resetRobot()
    env.render()
    time.sleep(1)
    #读取初始位置
    state=[*env.canvas.coords(env.rect1)[0:2],
               *env.canvas.coords(env.rect2)[0:2],
               *env.canvas.coords(env.rect3)[0:2],
               *env.canvas.coords(env.rect4)[0:2],
           *env.canvas.coords(env.rect5)[0:2],
           *env.canvas.coords(env.rect6)[0:2],
           *env.canvas.coords(env.rect7)[0:2],
           *env.canvas.coords(env.rect8)[0:2]]
    i = 1
    while True:
        n_act = nash_learn(state)
        state1 = move(n_act)
        obj, u1, u2, u3, u4, u5, u6, u7, u8 = compute_obj(state1)
        writer.add_scalar('ut', obj, i)
        writer.add_scalar('u1', u1, i)
        writer.add_scalar('u2', u2, i)
        writer.add_scalar('u3', u3, i)
        writer.add_scalar('u4', u4, i)
        writer.add_scalar('u5', u5, i)
        writer.add_scalar('u6', u6, i)
        writer.add_scalar('u7', u7, i)
        writer.add_scalar('u8', u8, i)
        i = i + 1
        path_1 = state1 + np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        env.render()
        n_act = nash_learn(state1)
        state = move(n_act)
        obj, u1, u2, u3, u4, u5, u6, u7, u8 = compute_obj(state)
        writer.add_scalar('ut', obj, i)
        writer.add_scalar('u1', u1, i)
        writer.add_scalar('u2', u2, i)
        writer.add_scalar('u3', u3, i)
        writer.add_scalar('u4', u4, i)
        writer.add_scalar('u5', u5, i)
        writer.add_scalar('u6', u6, i)
        writer.add_scalar('u7', u7, i)
        writer.add_scalar('u8', u8, i)
        i = i + 1
        path = state + np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        env.canvas.create_line(path_1[0], path_1[1], path[0], path[1], fill='SkyBlue1', width='2')
        env.canvas.create_line(path_1[2], path_1[3], path[2], path[3], fill='SkyBlue2', width='2')
        env.canvas.create_line(path_1[4], path_1[5], path[4], path[5], fill='RoyalBlue', width='2')
        env.canvas.create_line(path_1[6], path_1[7], path[6], path[7], fill='Midnightblue', width='2')
        env.canvas.create_line(path_1[8], path_1[9], path[8], path[9], fill='yellow', width='2')
        env.canvas.create_line(path_1[10], path_1[11], path[10], path[11], fill='gold', width='2')
        env.canvas.create_line(path_1[12], path_1[13], path[12], path[13], fill='gold3', width='2')
        env.canvas.create_line(path_1[14], path_1[15], path[14], path[15], fill='goldenrod', width='2')
if __name__ == '__main__':
    env.after(200,find_way())
    env.mainloop()










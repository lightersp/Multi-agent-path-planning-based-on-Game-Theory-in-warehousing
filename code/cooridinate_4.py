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

UNIT = 20  # 各自的最小长与宽
MAZE_H = 21  # 格子的高
MAZE_W = 21  # 格子的宽

# #起始点、目标点以及SVO
goal1=np.array([190, 330])
goal2=np.array([190, 50])
goal3=np.array([350,190])
goal4=np.array([50, 190])

origin1 = np.array([190, 50])
origin2 = np.array([190, 330])
origin3 = np.array([50, 190])
origin4 = np.array([350,190])

# svo=np.array([0,0,0,0])

# svo=np.array([np.pi/8,np.pi/8,np.pi/8,np.pi/8])

# svo=np.array([np.pi/4,np.pi/4,np.pi/4,np.pi/4])

svo=np.array([3*np.pi/8,3*np.pi/8,3*np.pi/8,3*np.pi/8])
writer=SummaryWriter('4_agent')
logFile1 = 'log_1.txt'  # 记录数据
logFile2 = 'log_2.txt'  # 记录数据
logFile3 = 'log_3.txt'  # 记录数据
logFile4 = 'log_4.txt'  # 记录数据
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.title('Warehouse')
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
        # #画路径点
        f = open(logFile1, 'r')
        s = f.readline()
        for count in range(56):
            s1=eval(f.readline())
            s2=eval(f.readline())
            self.canvas.create_line(s1[0][0] + 10, s1[0][1] + 10, s2[0][0] + 10, s2[0][1] + 10, fill='Red',width=4)
            self.canvas.create_line(s1[1][0] + 10, s1[1][1] + 10, s2[1][0] + 10, s2[1][1] + 10,fill='Red',width=4)
            self.canvas.create_line(s1[2][0] + 10, s1[2][1] + 10, s2[2][0] + 10, s2[2][1] + 10, fill='Red', width=4)
            self.canvas.create_line(s1[3][0] + 10, s1[3][1] + 10, s2[3][0] + 10, s2[3][1] + 10, fill='Red',width=4)
        f.close()
        f = open(logFile2, 'r')
        s = f.readline()
        for count in range(50):
            s1 = eval(f.readline())
            s2 = eval(f.readline())
            self.canvas.create_line(s1[0][0] + 10, s1[0][1] + 10, s2[0][0] + 10, s2[0][1] + 10, fill='Crimson',
                                    width=4)
            self.canvas.create_line(s1[1][0] + 10, s1[1][1] + 10, s2[1][0] + 10, s2[1][1] + 10, fill='Crimson',
                                    width=4)
            self.canvas.create_line(s1[2][0] + 10, s1[2][1] + 10, s2[2][0] + 10, s2[2][1] + 10, fill='Crimson',
                                    width=4)
            self.canvas.create_line(s1[3][0] + 10, s1[3][1] + 10, s2[3][0] + 10, s2[3][1] + 10, fill='Crimson',
                                    width=4)
        f.close()
        f = open(logFile3, 'r')
        s = f.readline()
        for count in range(50):
            s1 = eval(f.readline())
            s2 = eval(f.readline())
            self.canvas.create_line(s1[0][0] + 10, s1[0][1] + 10, s2[0][0] + 10, s2[0][1] + 10, fill='Brown',
                                    width=4)
            self.canvas.create_line(s1[1][0] + 10, s1[1][1] + 10, s2[1][0] + 10, s2[1][1] + 10, fill='Brown',
                                    width=4)
            self.canvas.create_line(s1[2][0] + 10, s1[2][1] + 10, s2[2][0] + 10, s2[2][1] + 10, fill='Brown',
                                    width=4)
            self.canvas.create_line(s1[3][0] + 10, s1[3][1] + 10, s2[3][0] + 10, s2[3][1] + 10, fill='Brown',
                                    width=4)
        f.close()
        f = open(logFile4, 'r')
        s = f.readline()
        for count in range(51):
            s1 = eval(f.readline())
            s2 = eval(f.readline())
            self.canvas.create_line(s1[0][0] + 10, s1[0][1] + 10, s2[0][0] + 10, s2[0][1] + 10, fill='DarkRed',
                                    width=4)
            self.canvas.create_line(s1[1][0] + 10, s1[1][1] + 10, s2[1][0] + 10, s2[1][1] + 10, fill='DarkRed',
                                    width=4)
            self.canvas.create_line(s1[2][0] + 10, s1[2][1] + 10, s2[2][0] + 10, s2[2][1] + 10, fill='DarkRed',
                                    width=4)
            self.canvas.create_line(s1[3][0] + 10, s1[3][1] + 10, s2[3][0] + 10, s2[3][1] + 10, fill='DarkRed',
                                    width=4)
        f.close()
        # 分拣站
        for i in range(1, 16, 7):
            self.canvas.create_rectangle(i * UNIT, 0, (i + 5) * UNIT, 2 * UNIT, fill='sandybrown')
        self.obstacle=[160, 160]
        #障碍物货架

        round_rectangle(160, 160, 240, 240, fill='seagreen',radius=70,width=1)

        # 目的地
        self.target1 = self.canvas.create_rectangle(
            190, 330, 210, 350,
            fill='SkyBlue1')
        self.target2 = self.canvas.create_rectangle(
            190, 50, 210,70,
            fill='SteelBlue2')
        self.target3 = self.canvas.create_rectangle(
            350,190, 370, 210,
            fill='RoyalBlue1')
        self.target4 = self.canvas.create_rectangle(
            50, 190, 70, 210,
            fill='Midnightblue')

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
        self.rect1 = round_rectangle(origin1[0] - 5, origin1[1] - 5,
                                   origin1[0] + 15, origin1[1] + 15,
                                   fill='SkyBlue1',radius=3,outline='black')
        self.rect2 = round_rectangle(origin2[0] - 5, origin2[1] - 5,
                                     origin2[0] + 15, origin2[1] + 15,
                                     fill='SteelBlue2', radius=3, outline='black')
        self.rect3 = round_rectangle(origin3[0] - 5, origin3[1] - 5,
                                     origin3[0] + 15, origin3[1] + 15,
                                     fill='RoyalBlue', radius=3, outline='black')
        self.rect4 = round_rectangle(origin4[0] - 5, origin4[1] - 5,
                                     origin4[0] + 15, origin4[1] + 15,
                                     fill='Midnightblue', radius=3, outline='black')
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2), self.canvas.coords(self.rect3),self.canvas.coords(self.rect4)
    #可视化渲染
    def render(self):
        time.sleep(0.01)
        self.update()
def  nash_learn(state):#通过casadi找最优控制
    T = 1  # 系统采样时间
    N = 3  # 需要预测的控制步长
    n_player=4#智能体数目
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

    goal=np.array([*goal1,*goal2,*goal3,*goal4])
        #各智能体到目标点的距离
    d_2_goal1 = ca.Function('d1', [position], [ca.norm_2(position[0:2] - goal[0:2])], ['input1'], ['d1'])
    d_2_goal2 = ca.Function('d2', [position], [ca.norm_2(position[2:4] - goal[2:4])], ['input2'], ['d2'])
    d_2_goal3 = ca.Function('d1', [position], [ca.norm_2(position[4:6] - goal[4:6])], ['input3'], ['d3'])
    d_2_goal4 = ca.Function('d1', [position], [ca.norm_2(position[6: ] - goal[6: ])], ['input4'], ['d4'])
        #各智能体之间的距离
    d_2_a1 = ca.Function('da1', [position], [ca.norm_2(position[0:2] - position[2:4]) +
                                             ca.norm_2(position[0:2] - position[4:6]) +
                                             ca.norm_2(position[0:2] - position[6:])], ['input11'], ['d11'])
    d_2_a2 = ca.Function('da2', [position], [ca.norm_2(position[2:4] - position[0:2]) +
                                             ca.norm_2(position[2:4] - position[4:6]) +
                                             ca.norm_2(position[2:4] - position[6:])], ['input12'], ['d12'])
    d_2_a3 = ca.Function('da3', [position], [ca.norm_2(position[4:6] - position[0:2]) +
                                             ca.norm_2(position[4:6] - position[2:4]) +
                                             ca.norm_2(position[4:6] - position[6:])], ['input13'], ['d13'])
    d_2_a4 = ca.Function('da4', [position], [ca.norm_2(position[6:] - position[0:2]) +
                                             ca.norm_2(position[6:] - position[2:4]) +
                                             ca.norm_2(position[6:] - position[4:6])], ['input14'], ['d14'])
        #总的奖励值
    reward1 = ca.Function('r1', [position], [d_2_goal1(position) - 0.1*d_2_a1(position)], ['agent_1_input'], ['agent_1_r'])
    reward2 = ca.Function('r2', [position], [d_2_goal2(position) - 0.1*d_2_a2(position)], ['agent_2_input'], ['agent_2_r'])
    reward3 = ca.Function('r3', [position], [d_2_goal3(position) - 0.1*d_2_a3(position)], ['agent_3_input'], ['agent_3_r'])
    reward4 = ca.Function('r4', [position], [d_2_goal4(position) - 0.1*d_2_a4(position)], ['agent_4_input'], ['agent_4_r'])
       #备用奖励函数以备不收敛的情况发生orz
    # reward1 = ca.Function('r1', [position], [d_2_goal1(position)], ['agent_1_input'], ['agent_1_r'])
    # reward2 = ca.Function('r2', [position], [d_2_goal2(position)], ['agent_2_input'], ['agent_2_r'])
    # reward3 = ca.Function('r3', [position], [d_2_goal3(position) ], ['agent_3_input'], ['agent_3_r'])
    # reward4 = ca.Function('r4', [position], [d_2_goal4(position) ], ['agent_4_input'], ['agent_4_r'])

    #即时效用函数

    u1=ca.Function('u1',[position],[(np.cos(svo[0])*reward1(position)+
                                    np.sin(svo[0])*reward2(position)+
                                    np.sin(svo[0])*reward3(position)+
                                    np.sin(svo[0])*reward4(position))/3],['utility1_input'],['utility1_output'])
    u2=ca.Function('u2',[position],[(np.cos(svo[1])*reward2(position)+
                                    np.sin(svo[1])*reward1(position)+
                                    np.sin(svo[1])*reward3(position)+
                                    np.sin(svo[1])*reward4(position))/3],['utility2_input'],['utility2_output'])
    u3=ca.Function('u3',[position],[(np.cos(svo[2])*reward3(position)+
                                    np.sin(svo[2])*reward2(position)+
                                    np.sin(svo[2])*reward1(position)+
                                    np.sin(svo[2])*reward4(position))/3],['utility3_input'],['utility3_output'])
    u4=ca.Function('u4',[position],[(np.cos(svo[3])*reward4(position)+
                                    np.sin(svo[3])*reward2(position)+
                                    np.sin(svo[3])*reward3(position)+
                                    np.sin(svo[3])*reward1(position))/3],['utility4_input'],['utility4_output'])
    #总效用
    u1_total = 0;u2_total=0;u3_total=0;u4_total=0
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
    obj=-u1_total - u2_total - u3_total - u4_total
    #约束条件
     ##状态限制,不能出界
    g=[]
    ubg=[]
    lbg=[]
    for inter1 in range(n_player*n_state):
         for inter2 in range(N):
             g.append(x[inter1,inter2])
             ubg.append(20*20)
             lbg.append(0)
     ##智能体间的距离
    interval=np.array([10,10])#智能体间的中心偏移
    interval2=np.array(([40,40]))#智能体与货架间的中心偏移
    for inter3 in range(1,4):
        g.append(ca.norm_2((x[0:2,inter3] + interval) - (x[2:4,inter3] + interval))-20)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[4:6,inter3] + interval))-20)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (x[6:,inter3] + interval))-20)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[4:6,inter3] + interval))-20)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (x[6:,inter3] + interval))-20)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[4:6, inter3] + interval) - (x[6:,inter3] + interval))-20)
        lbg.append(0)
        ubg.append(np.inf)
        #与障碍物的距离
        g.append(ca.norm_2((x[0:2, inter3] + interval) - (env.obstacle + interval2))-70)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[2:4, inter3] + interval) - (env.obstacle + interval2))-70)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[4:6, inter3] + interval) - (env.obstacle + interval2))-70)
        lbg.append(0)
        ubg.append(np.inf)
        g.append(ca.norm_2((x[6:, inter3] + interval) - (env.obstacle + interval2))-70)
        lbg.append(0)
        ubg.append(np.inf)

    lbx=-3
    ubx=3

    nlp = {'f': -obj, 'x': ca.reshape(u, -1, 1),'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp,opts_setting)

    r = solver(x0=np.zeros((n_control*n_player,N)).reshape(-1,1),lbx=lbx,ubx=ubx,ubg=ubg,lbg=lbg)
    best_act=r['x']
    act_next=best_act[0:8]
    return act_next





def compute_obj(state):
    state=np.array(state)
    goal = np.array([*goal1, *goal2, *goal3, *goal4])
    # 各智能体到目标点的距离
    d_2_goal1 = np.linalg.norm(state[0:2] - goal[0:2])
    d_2_goal2 = np.linalg.norm(state[2:4] - goal[2:4])
    d_2_goal3 = np.linalg.norm(state[4:6] - goal[4:6])
    d_2_goal4 = np.linalg.norm(state[6:] - goal[6:])
    # 各智能体之间的距离
    d_2_a1 = np.linalg.norm(state[0:2] - state[2:4]) +np.linalg.norm(state[0:2] - state[4:6]) +np.linalg.norm(state[0:2] - state[6:])
    d_2_a2 = np.linalg.norm(state[2:4] - state[0:2]) +np.linalg.norm(state[2:4] - state[4:6]) +np.linalg.norm(state[2:4] - state[6:])
    d_2_a3 = np.linalg.norm(state[4:6] - state[0:2]) +np.linalg.norm(state[4:6] - state[2:4]) +np.linalg.norm(state[4:6] - state[6:])
    d_2_a4 = np.linalg.norm(state[6:] - state[0:2]) +np.linalg.norm(state[6:] - state[2:4]) +np.linalg.norm(state[6:] - state[4:6])
    # 总的奖励值
    reward1 = d_2_goal1- 0.1 * d_2_a1
    reward2 = d_2_goal2 - 0.1 * d_2_a2
    reward3 = d_2_goal3- 0.1 * d_2_a3
    reward4 = d_2_goal4 - 0.1 * d_2_a4


    u1 = (np.cos(svo[0]) * reward1 +np.sin(svo[0]) * reward2 +np.sin(svo[0]) * reward3 +np.sin(svo[0]) * reward4) / 3
    u2 = (np.cos(svo[1]) * reward2 +np.sin(svo[1]) * reward1 +np.sin(svo[1]) * reward3 +np.sin(svo[1]) * reward4) / 3
    u3 = (np.cos(svo[2]) * reward3 +np.sin(svo[2]) * reward2 +np.sin(svo[2]) * reward1 +np.sin(svo[2]) * reward4) / 3
    u4 = (np.cos(svo[3]) * reward4 +np.sin(svo[3]) * reward2 +np.sin(svo[3]) * reward3 +np.sin(svo[3]) * reward1) / 3

    obj=-u1 - u2 - u3 - u4
    return obj,-u1,-u2,-u3,-u4
def move(action):
    env.canvas.move(env.rect1, action[0],action[1])
    env.canvas.move(env.rect2, action[2],action[3])
    env.canvas.move(env.rect3, action[4],action[5])
    env.canvas.move(env.rect4, action[6],action[7])
    state = [*env.canvas.coords(env.rect1)[0:2],
             *env.canvas.coords(env.rect2)[0:2],
             *env.canvas.coords(env.rect3)[0:2],
             *env.canvas.coords(env.rect4)[0:2]]
    print('agent1{0}agent2{1}agent3{2}agent4{3}'.format(state[:2],state[2:4],state[4:6],state[6:]))
    return state

env = Maze()
def find_way():
    #复位
    env.resetRobot()
    env.render()
    time.sleep(3)
    i=1
    #读取初始位置
    state=[*env.canvas.coords(env.rect1)[0:2],
               *env.canvas.coords(env.rect2)[0:2],
               *env.canvas.coords(env.rect3)[0:2],
               *env.canvas.coords(env.rect4)[0:2]]
    while (i<200):
        n_act=nash_learn(state)
        state=move(n_act)
        obj,u1,u2,u3,u4=compute_obj(state)
        writer.add_scalar('total_obj', obj, i)
        writer.add_scalar('utility_1', u1, i)
        writer.add_scalar('utility_2', u2, i)
        writer.add_scalar('utility_3', u3, i)
        writer.add_scalar('utility_4', u4, i)
        i=i+1
        env.render()
        time.sleep(0.5)
        n_act = nash_learn(state)
        state = move(n_act)
        obj, u1, u2, u3, u4 = compute_obj(state)
        writer.add_scalar('total_obj', obj, i)
        writer.add_scalar('utility_1', u1, i)
        writer.add_scalar('utility_2', u2, i)
        writer.add_scalar('utility_3', u3, i)
        writer.add_scalar('utility_4', u4, i)
        i=i+1
if __name__ == '__main__':
    env.after(200,find_way())
    env.mainloop()










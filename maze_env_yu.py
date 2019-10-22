import numpy as np
import time
import tkinter as tk
import random

UNIT = 40  # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width

# env1
# GOAL = np.array([4, 1])
# BLOCK = np.array([[2, 1], [3, 1], [2, 3], [3, 3]])
# INITIAL = np.array([[1, 1],[1, 2],[1, 3], [1, 4]])

# env2
# GOAL = np.array([1, 5])
# BLOCK = np.array([[1, 3],[2, 3],[3, 3], [4, 3]])
# INITIAL = np.array([[1, 1],[1, 2],[2, 1], [2, 2],[3, 1],[3, 2]])

# env3
GOAL = np.array([4, 2])
BLOCK = np.array([[1, 3],[3, 3],[3, 2], [3, 1]])
INITIAL = np.array([[1, 1],[1, 2],[2, 1], [2, 2]])

AGENT = np.array([1, 1])


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2

        self.origin = np.array([UNIT / 2, UNIT / 2])

        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))

        self.maze_h = MAZE_H
        self.maze_w = MAZE_W

        self.goal = GOAL
        self.block = BLOCK
        self.agent = AGENT
        self.initial = INITIAL
        # self.map = np.zeros((MAZE_W, MAZE_H))

        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create goal
        goal_center = self.origin + np.array([UNIT * (self.goal[0] - 1),
                                              UNIT * (self.goal[1] - 1)])
        self.goal_cvs = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

        # create blocks
        self.block_cvs = []
        for bl in self.block:
            block_center = self.origin + np.array([UNIT * (bl[0] - 1),
                                                   UNIT * (bl[1] - 1)])
            block_cvs_iter = self.canvas.create_rectangle(
                block_center[0] - 15, block_center[1] - 15,
                block_center[0] + 15, block_center[1] + 15,
                fill='black')
            self.block_cvs.append(block_cvs_iter)

        # pack all
        self.canvas.pack()

    def reset(self, agent_cor=None):
        if agent_cor is None:
            self.agent = np.array([1, 1])
        else:
            self.agent = agent_cor

        # return state
        return np.array([self.agent[0] / self.maze_w - 0.625,
                         self.agent[1] / self.maze_h - 0.625])

    def step(self, action):
        hit = False
        if action == 0:  # up
            updated_cor = self.agent + np.array([0, -1])
            if updated_cor[1] >= 1 and (updated_cor.tolist()
                                        not in self.block.tolist()):
                self.agent = updated_cor
            else:
                hit = True
        elif action == 1:  # down
            updated_cor = self.agent + np.array([0, 1])
            if updated_cor[1] <= self.maze_h and (updated_cor.tolist()
                                        not in self.block.tolist()):
                self.agent = updated_cor
            else:
                hit = True
        elif action == 2:  # right
            updated_cor = self.agent + np.array([1, 0])
            if updated_cor[0] <= self.maze_w and (updated_cor.tolist()
                                        not in self.block.tolist()):
                self.agent = updated_cor
            else:
                hit = True
        elif action == 3:  # left
            updated_cor = self.agent + np.array([-1, 0])
            if updated_cor[0] >= 1 and (updated_cor.tolist()
                                        not in self.block.tolist()):
                self.agent = updated_cor
            else:
                hit = True

        # reward function
        if np.array_equal(self.agent, self.goal):
            reward = 10
            done = True
        else:
            reward = 0
            done = False

        if hit:
            reward -= 0.1

        # next state
        s_ = np.array([self.agent[0] / self.maze_w - 0.625,
                       self.agent[1] / self.maze_h - 0.625])
        return s_, reward, done

    def render(self):
        try:
            self.canvas.delete(self.agent_cvs)
        except:
            pass
        # render agent
        agent_center = self.origin + np.array([UNIT * (self.agent[0] - 1),
                                               UNIT * (self.agent[1] - 1)])
        self.agent_cvs = self.canvas.create_oval(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

        self.update()
        time.sleep(0.1)

    def random_pos(self):

        while True:
            pos = random.choice(self.initial)
            if (pos.tolist() not in self.goal.tolist()) and (pos.tolist() not in self.block.tolist()):
                break

        return pos




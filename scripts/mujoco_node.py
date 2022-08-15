#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import radians

import subprocess
import rospy
import rospkg

class mj_node():
    def __init__(self):
        path='/home/beau/Documents/ws_moveit/src/HIL'
        name=path+'/assets/ur3e/ur3e_dislodge.xml'
        # name=path+'/assets/ur3e/ur3e_wallv2.xml'
        model  = load_model_from_path(name)
        self.sim = MjSim(model)

        # self.init_joints=np.array([1.6382425937090916, 0.4256844286749071, -1.2910630181298555, -2.280060051656477, -0.06818132784290483, -1.5666176622613293])
        # self.init_joints=np.array([-0.10243801853235805, 0.6645765466505216, -2.125822600714074, -1.6799365152523613, 0.8937045971511799, -1.571226426532903])
        self.init_joints=np.array([radians(150.19), radians(-24.29), radians(63.94), radians(-31.8), radians(-221.13), radians(75)])
        for i in range(len(self.init_joints)):
            self.sim.data.qpos[i]=self.init_joints[i]
        self.sim.forward()

        self.ee_index=self.sim.model.site_name2id('mep:ee')
        ind= [0,1,2,3,4,5]
        self.joint_index={'joints': ind, 'qpos': ind, 'qvel': ind}


        




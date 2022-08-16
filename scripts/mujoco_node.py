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
        # name=path+'/assets/ur16_test.xml'#env.xml'
        name=path+'/assets/env.xml'
        # name=path+'/assets/ur3e/ur3e_wallv2.xml'
        model  = load_model_from_path(name)
        self.sim = MjSim(model)

        # self.init_joints=np.array([1.6382425937090916, 0.4256844286749071, -1.2910630181298555, -2.280060051656477, -0.06818132784290483, -1.5666176622613293])
        # self.init_joints=np.array([2.62130915,  0.05102677, -2.36490028,  2.32357941, -5.1172589 , 1.56649076])
        self.init_joints=np.array([3.60904057414704, -1.6234970535894417, 2.015301533456138, 2.749788173723113, -2.0382442473521483, -4.712388980384706, -0.080488600775479, -1.5729014582396508, -1.9967997749623725, 3.56969627204389, -1.6513264074081029, 1.570795912851109])
        
        # self.init_joints=np.array([radians(150.19), radians(-24.29), radians(63.94), radians(-31.8), radians(-221.13), radians(75)])
        for i in range(len(self.init_joints)):
            self.sim.data.qpos[i]=self.init_joints[i]
        self.sim.forward()

        # self.ee_index=self.sim.model.site_name2id('mep:ee')
        # ind= [0,1,2,3,4,5]
        # self.joint_index={'joints': ind, 'qpos': ind, 'qvel': ind}


        




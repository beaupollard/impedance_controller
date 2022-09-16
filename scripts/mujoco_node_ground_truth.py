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
        path='/home/biorobotics/Documents/impedance_ctrl-main'
        # name=path+'/assets/ur16_test.xml'#env.xml'
        # name=path+'/assets/env.xml'
        # name=path+'/assets/env_groundtruth.xml'
        name=path+'/assets/env_force.xml'
        model  = load_model_from_path(name)
        self.sim = MjSim(model)


        # self.init_joints=np.array([-0.1968898744057256, -2.362684835408132, -1.9450582591928847, 4.307743102878401, -1.767686224021063, 1.570796328411924])# UR16 ground truth
        # self.init_joints=np.array([1.3285446, -2.21587002, -2.45253666, -1.6461946, 4.51778477, -3.24643225e-04])# UR16 ground truth
        self.init_joints=np.array([1.87738946,-0.67772296, 0.76914369, -0.09152432, 1.87700289, 3.13330789])# UR3 force input

        for i in range(len(self.init_joints)):
            self.sim.data.qpos[i]=self.init_joints[i]
        self.sim.forward()

        # self.ee_index=self.sim.model.site_name2id('mep:ee')
        # ind= [0,1,2,3,4,5]
        # self.joint_index={'joints': ind, 'qpos': ind, 'qvel': ind}


        

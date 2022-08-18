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
        # name=path+'/assets/env_groundtruth.xml'
        # name=path+'/assets/env_groundtruth_wall.xml'
        model  = load_model_from_path(name)
        self.sim = MjSim(model)

        # self.init_joints=np.array([1.6382425937090916, 0.4256844286749071, -1.2910630181298555, -2.280060051656477, -0.06818132784290483, -1.5666176622613293])
        # self.init_joints=np.array([2.62130915,  0.05102677, -2.36490028,  2.32357941, -5.1172589 , 1.56649076])
        
        # self.init_joints=np.array([-0.1630317064275968, -2.458356132275945, -1.6527857916516833, 4.111142690777924, -1.7338293174042148, 1.5707964510262307])# UR16 ground truth
        self.init_joints=np.array([3.6009982720913998, -0.12609548703807621, 1.2539939019860429, 2.013694239353162, -2.0302019664497255, -4.712388975115711, -0.1968898744057256, -2.362684835408132, -1.9450582591928847, 4.307743102878401, -1.767686224021063, 1.570796328411924])
        
        # self.init_joints=np.array([2.3007798447900876, 0.5572989086360547, -1.3570023284546566, 0.799702748035126, 0.20641476926149863, 1.5707969843231002]) #UR16 ground truth wall
        for i in range(len(self.init_joints)):
            self.sim.data.qpos[i]=self.init_joints[i]
        self.sim.forward()

        # self.ee_index=self.sim.model.site_name2id('mep:ee')
        # ind= [0,1,2,3,4,5]
        # self.joint_index={'joints': ind, 'qpos': ind, 'qvel': ind}


        




#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mujoco_node import mj_node
from scipy.spatial.transform import Rotation as R
from robosuite.controllers.osc import OperationalSpaceController as osc
from qp_opt import qp_opt
import math
from utils import butter_lowpass_filter
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from ros_utils import ur_controllers
import time

## Setup subscripers and publishers ##
ctrl=ur_controllers()
rospy.init_node('ros_ctrl') 
pub = rospy.Publisher("/left/ur_hardware_interface/script_command", String, queue_size=1)
substates = rospy.Subscriber("/left/joint_states", JointState, ctrl.jstate_in)
# subpub = rospy.Subscriber("/ur_hardware_interface/script_command", String, ctrl.msgs_published)
commands = []

## Setup Mujoco Sim Node ##
mj_sim=mj_node()

view=True
if view==True:
    viewer=MjViewer(mj_sim.sim)

## Setup Controller ##
F_des=np.array([0,0,0,0,0,0],dtype=np.float64)     # Desired force in ee frame
qp=qp_opt(mj_sim.sim,F_des=F_des,optimize=True,hybrid=False)

while pub.get_num_connections()==0:
    time.sleep(0.5)

ctrl.movej(mj_sim.init_joints,pub)
commands.append(mj_sim.init_joints)
time.sleep(2)



q_opt=[]
tau_0=[]
count=0
while mj_sim.sim.data.time<10:

    # ## Run the QP optimizer ##
    q_out=qp.run_opt()

    ## Step Mujoco Forward ##
    mj_sim.sim.step()

    q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
    tau_0.append([pos for pos in mj_sim.sim.data.sensordata[:3]])
    ctrl.servoj(mj_sim.sim.data.qpos[:6],pub)
    # if count==2:
    #     # if np.max(abs(mj_sim.sim.data.qpos[:6]-ctrl.jstate))<0.05:
    #     # ctrl.servoj(mj_sim.sim.data.qpos[:6],pub)
    #     commands.append(mj_sim.sim.data.qpos[:6])
    #     # ctrl.servoj(mj_sim.sim.data.qpos[:6],pub)
    #     # while np.max(abs(mj_sim.sim.data.qpos[:6]-ctrl.jstate))>0.05:
    #     #     pass

    #     count=0
    # else:
    #     count+=1

    if view==True:
        viewer.render()

t0=np.array(tau_0)
q0=np.array(q_opt)*180/math.pi
np.savez('./commands.npz',commands=commands)

y=butter_lowpass_filter(t0[:,2],10,1/mj_sim.sim.model.opt.timestep,5)
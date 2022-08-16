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

## Setup Mujoco Sim Node ##
mj_sim=mj_node()

view=True
if view==True:
    viewer=MjViewer(mj_sim.sim)

## Setup Controller ##
F_des=np.array([0,0,0,0,0,0],dtype=np.float64)     # Desired force in ee frame
qp_ur3=qp_opt(mj_sim.sim,F_des=F_des,optimize=False,hybrid=False,ee_site='ur3:ee',target_site='wpt1',qvel_index=np.array([0,1,2,3,4,5]),peg_name=['peg_base'])
qp_ur16=qp_opt(mj_sim.sim,F_des=F_des,optimize=False,hybrid=False,ee_site='ur16:ee',target_site='wpt2',qvel_index=6+np.array([0,1,2,3,4,5]),peg_name=['ur16_peg'])
# qp_ur3.kp=np.array([120,120,120,120,120,120])
# qp_ur3.kd=np.array([100,100,100,100,100,100])
qp_ur16.kp=np.array([300,300,300,300,300,300])
qp_ur16.kd=np.array([200,200,200,200,200,200])
q_opt=[]
tau_0=[]
while mj_sim.sim.data.time<20:
    # mj_sim.sim.data.ctrl[:]=0
    # ## Run the QP optimizer ##
    q_out=qp_ur3.run_opt()
    q_out2=qp_ur16.run_opt()

    ## Step Mujoco Forward ##
    mj_sim.sim.step()

    q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
    tau_0.append([pos for pos in mj_sim.sim.data.sensordata[:3]])

    if view==True:
        viewer.render()

t0=np.array(tau_0)
q0=np.array(q_opt)*180/math.pi

y=butter_lowpass_filter(t0[:,2],10,1/mj_sim.sim.model.opt.timestep,5)
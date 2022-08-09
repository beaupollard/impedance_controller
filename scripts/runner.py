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
F_des=np.array([0,0,2,0,0,0],dtype=np.float64)     # Desired force in ee frame
qp=qp_opt(mj_sim.sim,F_des=F_des,optimize=True)

q_opt=[]
tau_0=[]
while mj_sim.sim.data.time<15:

    # ## Run the QP optimizer ##
    q_out=qp.run_opt()

    ## Step Mujoco Forward ##
    mj_sim.sim.step()

    q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
    tau_0.append([pos for pos in mj_sim.sim.data.sensordata[:3]])

    if view==True:
        viewer.render()

t0=np.array(tau_0)
q0=np.array(q_opt)*180/math.pi

y=butter_lowpass_filter(t0[:,2],10,1/mj_sim.sim.model.opt.timestep,5)
#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mujoco_node import mj_node
from scipy.spatial.transform import Rotation as R
from robosuite.controllers.osc import OperationalSpaceController as osc
from qp_opt import qp_opt

## Setup Mujoco Sim Node ##
mj_sim=mj_node()

view=True
if view==True:
    viewer=MjViewer(mj_sim.sim)

## Setup Controller ##
qp=qp_opt(mj_sim.sim)

q_opt=[]
tau_0=[]
while mj_sim.sim.data.time<10:

    # ## Run the QP optimizer ##
    q_out=qp.run_opt()

    # q_out=qp.run_opt(mj_sim.sim.data.qacc)
    # q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
    # tau_0.append([pos for pos in tau])

    ## Step Mujoco Forward ##
    mj_sim.sim.step()

    if view==True:
        viewer.render()

t0=np.array(tau_0)
q0=np.array(q_opt)
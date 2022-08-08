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

## Setup the controller ##
ctrl=osc(sim=mj_sim.sim,eef_name='mep:ee',joint_indexes=mj_sim.joint_index,actuator_range=(-100*np.ones(6),100*np.ones(6)))
ctrl.kp=120*np.ones(6)
ctrl.kd=60.*np.ones(6)
ctrl.model_timestep=mj_sim.sim.model.opt.timestep
constraints=[]
dt=mj_sim.sim.model.opt.timestep
qp=qp_opt(mj_sim.sim)
q_opt=[]
tau_0=[]
while mj_sim.sim.data.time<10:
    r=R.from_matrix(mj_sim.sim.data.get_body_xmat('satellite'))
    r2=R.from_matrix(mj_sim.sim.data.get_body_xmat('antenna'))
    ctrl.goal_pos=mj_sim.sim.data.get_site_xpos('wpt1')
    ctrl.goal_ori=r2.as_matrix()@r.as_matrix()

    t0=mj_sim.sim.data.time
    tau, J_full, decoupled_wrench, torque_comp =ctrl.run_controller()
    
    ## Set the calculated torques
    for i in range(6):
        mj_sim.sim.data.ctrl[i]=tau[i]

    ## Determine the joint accelerations ##
    mj_sim.sim.forward()

    ## Run the QP optimizer ##
    q_out=qp.run_opt(mj_sim.sim.data.qacc)
    q_opt.append([pos for pos in q_out])
    tau_0.append([pos for pos in tau])

    ## Step Mujoco Forward ##
    mj_sim.sim.step()
    if view==True:
        viewer.render()

t0=np.array(tau_0)
q0=np.array(q_opt)
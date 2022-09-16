#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer, generated
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mujoco_node_ground_truth import mj_node
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
from geometry_msgs.msg import WrenchStamped
sensor_data_point = 0
force_x_array = []
force_y_array = []
force_z_array = []


def sensor_callback(data):
    global sensor_data_point
    sensor_data_point = data.wrench.force


## Setup subscripers and publishers ##
ctrl=ur_controllers()
rospy.init_node('ros_ctrl') 
# ur3_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=1)
# ur3_substates = rospy.Subscriber("/joint_states", JointState, ctrl.jstate_in)
ur16_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=1)
ur16_substates = rospy.Subscriber("/joint_states", JointState, ctrl.jstate_in)
sensorSub = rospy.Subscriber('/netft_data', WrenchStamped, sensor_callback)

# subpub = rospy.Subscriber("/ur_hardware_interface/script_command", String, ctrl.msgs_published)
commands = []

## Setup Mujoco Sim Node ##
mj_sim=mj_node()

view=True
if view==True:
    viewer=MjViewer(mj_sim.sim)
    viewer.cam.type = generated.const.CAMERA_FIXED
    viewer.cam.fixedcamid = 0 

## Setup Controller ##
F_des=np.array([0,0,17.6,0,0,0],dtype=np.float64)
nozzle_name=['peg_base','nozzle','nozzle0','nozzle45','nozzle90','nozzle135','nozzle180','nozzle225','nozzle270','nozzle315']
peg_name=['ur16_pegbase','ur16_peg']
qp_ur16=qp_opt(mj_sim.sim,F_des=F_des,optimize=False,hybrid=False,ee_site='ur16:ee',target_site='wpt2',qvel_index=np.array([0,1,2,3,4,5]),peg_name=peg_name)
qp_ur16.kp=np.array([300,300,300,300,300,300])
qp_ur16.kd=np.array([200,200,200,200,200,200])
qp_ur16.acc_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('ur16_acc')]
qp_ur16.force_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('ur16_force_sensor')]
qp_ur16.torque_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('ur16_torque_sensor')]
force_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('force_sensor')]

# while ur16_pub.get_num_connections()==0:
#     time.sleep(0.5)


ctrl.movej(mj_sim.init_joints[:6],ur16_pub)
commands.append(mj_sim.init_joints)
time.sleep(2)



q_opt=[]
tau_0=[]
count=0
while mj_sim.sim.data.time<5:

    # ## Run the QP optimizer ##
    q_out2=qp_ur16.run_opt()
    ## Step Mujoco Forward ##
    mj_sim.sim.step()

    q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
    tau_0.append([pos for pos in mj_sim.sim.data.sensordata[force_index:force_index+3]])
    print(mj_sim.sim.data.qpos[:6])
    ctrl.servoj(mj_sim.sim.data.qpos[:6],ur16_pub)
    force_x_array.append(sensor_data_point.x)
    force_y_array.append(sensor_data_point.y)
    force_z_array.append(sensor_data_point.z)
    # ctrl.servoj(mj_sim.sim.data.qpos[6:],ur16_pub)
    if np.max(abs(mj_sim.sim.data.qpos[:6]-ctrl.jstate))<0.05:
        ctrl.servoj(mj_sim.sim.data.qpos[:6],ur16_pub)
        # print(mj_sim.sim.data.qpos[:6])
        commands.append(mj_sim.sim.data.qpos[:6])
        ctrl.servoj(mj_sim.sim.data.qpos[:6],ur16_pub)
        while np.max(abs(mj_sim.sim.data.qpos[:6]-ctrl.jstate))>0.05:
            pass

    if view==True:
        viewer.render()


plt.plot(force_x_array, 'r')
plt.plot(force_y_array, 'g')
plt.plot(force_z_array,'b')
plt.legend(['X','Y','Z'])
plt.show()
ctrl.movej(mj_sim.init_joints[:6],ur16_pub)
commands.append(mj_sim.init_joints)
t0=np.array(tau_0)
q0=np.array(q_opt)*180/math.pi  
np.savez('./force.npz', x = force_x_array, y = force_y_array,  z = force_z_array)
y=butter_lowpass_filter(t0[:,2],10,1/mj_sim.sim.model.opt.timestep,5)
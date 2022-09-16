#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer, generated
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
from geometry_msgs.msg import WrenchStamped
import time
import tf

sensor_array = np.zeros(6)
sensor_data_point = 0
stater = []
position = []
position_ur = []



def sensor_callback(data):
    global sensor_data_point
    sensor_data_point = data.wrench
    sensor_array[0] = sensor_data_point.force.x + 1.396185
    sensor_array[1] = sensor_data_point.force.y + 1.889446
    sensor_array[2] = sensor_data_point.force.z + 4.168303
    sensor_array[3] = sensor_data_point.torque.x + 0.095789
    sensor_array[4] = sensor_data_point.torque.y - 0.100252
    sensor_array[5] = sensor_data_point.torque.z - 0.326551



## Setup subscripers and publishers ##
ctrl=ur_controllers()
rospy.init_node('ros_ctrl') 
r = rospy.Rate(100) 
ur3_pub = rospy.Publisher("/left/ur_hardware_interface/script_command", String, queue_size=1)
ur16_pub = rospy.Publisher("/right/ur_hardware_interface/script_command", String, queue_size=1)
ur3_substates = rospy.Subscriber("/left/joint_states", JointState, ctrl.jstate_in_left)
ur16_substates = rospy.Subscriber("/right/joint_states", JointState, ctrl.jstate_in_right)
sensorSub = rospy.Subscriber('/netft_data', WrenchStamped, sensor_callback)


# subpub = rospy.Subscriber("/ur_hardware_interface/script_command", String, ctrl.msgs_published)
commands = []

listener = tf.TransformListener()


## Setup Mujoco Sim Node ##
mj_sim=mj_node()

view=True
if view==True:
    viewer=MjViewer(mj_sim.sim)
    # viewer.cam.type = generated.const.CAMERA_FIXED
    # viewer.cam.fixedcamid = 0 

## Setup Controller ##
F_des=np.array([0,0,0,0,0,0],dtype=np.float64)
nozzle_name=['peg_base','nozzle','nozzle0','nozzle45','nozzle90','nozzle135','nozzle180','nozzle225','nozzle270','nozzle315']
peg_name=['ur16_pegbase','ur16_peg']
qp_ur3=qp_opt(mj_sim.sim,F_des=F_des,optimize=False,hybrid=False,ee_site='ur3:ee',target_site='wpt1',qvel_index=np.array([0,1,2,3,4,5]),peg_name=nozzle_name)
qp_ur16=qp_opt(mj_sim.sim,F_des=F_des,optimize=False,hybrid=False,ee_site='ur16:ee',target_site='wpt2',qvel_index=6+np.array([0,1,2,3,4,5]),peg_name=peg_name)
qp_ur3.kp=np.array([80,40,60,60,60,60])
qp_ur3.kd=np.array([80,30,60,60,60,60])
qp_ur16.kp=np.array([60,60,60,60,60,60])*1.5
qp_ur16.kd=np.array([60,60,60,60,60,60])*1.5
qp_ur16.acc_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('ur16_acc')]
qp_ur16.force_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('ur16_force_sensor')]
qp_ur16.torque_index=mj_sim.sim.model.sensor_adr[mj_sim.sim.model.sensor_name2id('ur16_torque_sensor')]

while ur3_pub.get_num_connections()==0:
    time.sleep(0.5)

while ur16_pub.get_num_connections()==0:
    time.sleep(0.5)


# while ur16_pub.get_num_connections()==0:
#     time.sleep(0.5)

ctrl.movej(mj_sim.init_joints[:6],ur3_pub)
ctrl.movej(mj_sim.init_joints[6:],ur16_pub)
# commands.append(mj_sim.init_joints)
# time.sleep(2)



q_opt=[]
tau_0=[]
count=0
print('starting simulation')
while mj_sim.sim.data.time<10:
    if abs(sensor_array[2]) > 15:
        print('force too high, exiting')
        break
    else:
        # ## Run the QP optimizer ##
        q_out=qp_ur3.run_opt(sensor_array)
        q_out2=qp_ur16.run_opt(sensor_array)

        ## Step Mujoco Forward ##
        mj_sim.sim.step()

        q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
        tau_0.append([pos for pos in mj_sim.sim.data.sensordata[:3]])
        ctrl.servoj(mj_sim.sim.data.qpos[:6],ur3_pub)
        ctrl.servo16j(mj_sim.sim.data.qpos[6:],ur16_pub)

        try:
            (transl,rotl) = listener.lookupTransform('leftbase', 'lefttool0_controller', rospy.Time(0))
            (transr,rotr) = listener.lookupTransform('rightbase', 'righttool0_controller', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        r.sleep()
        if view==True:
            viewer.render()

print('simulation complete')
print("Ur3e")
print(mj_sim.sim.data.qpos[:6])
print(mj_sim.sim.data.site_xpos[mj_sim.sim.model.site_name2id('ur3:ee')])
# print(transl)
print("UR16e")
print(mj_sim.sim.data.qpos[6:])
print(mj_sim.sim.data.site_xpos[mj_sim.sim.model.site_name2id('ur16:ee')])
# print(transr)
t0=np.array(tau_0)
q0=np.array(q_opt)*180/math.pi
# np.savez('./commands.npz',commands=commands)

# y=butter_lowpass_filter(t0[:,2],10,1/mj_sim.sim.model.opt.timestep,5)
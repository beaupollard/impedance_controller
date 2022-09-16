#!/usr/bin/env python3
from cProfile import label
from turtle import pos
from mujoco_py import load_model_from_path, MjSim, MjViewer, generated
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from mujoco_node_ground_truth import mj_node
from scipy.spatial.transform import Rotation as R
from robosuite.controllers.osc import OperationalSpaceController as osc
from qp_opt import qp_opt
import math
from utils import butter_lowpass_filter
import rospy
import tf

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from ros_utils import ur_controllers
import time
from geometry_msgs.msg import WrenchStamped
from copy import deepcopy

sensor_array = np.zeros(6)
sensor_data_point = 0
stater = []
position = []
position_ur = []



def sensor_callback(data):
    global sensor_data_point
    sensor_data_point = data.wrench
    sensor_array[0] = sensor_data_point.force.x + 1.7473049999999999
    sensor_array[1] = sensor_data_point.force.y + 1.663284
    sensor_array[2] = sensor_data_point.force.z + 3.245171
    sensor_array[3] = sensor_data_point.torque.x + 0.107929
    sensor_array[4] = sensor_data_point.torque.y - 0.107058
    sensor_array[5] = sensor_data_point.torque.z - 0.32332299999999997

    


## Setup subscripers and publishers ##
ctrl=ur_controllers()
rospy.init_node('ros_ctrl') 
r = rospy.Rate(100) 
ur3_pub = rospy.Publisher("/left/ur_hardware_interface/script_command", String, queue_size=1)
ur16_pub = rospy.Publisher("/right/ur_hardware_interface/script_command", String, queue_size=1)
ur3_substates = rospy.Subscriber("/left/joint_states", JointState, ctrl.jstate_in)
ur16_substates = rospy.Subscriber("/right/joint_states", JointState, ctrl.jstate_in)
sensorSub = rospy.Subscriber('/netft_data', WrenchStamped, sensor_callback)


listener = tf.TransformListener()


# subpub = rospy.Subscriber("/ur_hardware_interface/script_command", String, ctrl.msgs_published)
# commands = []

## Setup Mujoco Sim Node ##
mj_sim=mj_node()

view=False
if view==True:
    viewer=MjViewer(mj_sim.sim)
    # viewer.cam.type = generated.const.CAMERA_FIXED
    # viewer.cam.fixedcamid = 0 

## Setup Controller ##
F_des=np.array([0,0,12,0,0,0],dtype=np.float64)
nozzle_name=['peg_base','nozzle','nozzle0','nozzle45','nozzle90','nozzle135','nozzle180','nozzle225','nozzle270','nozzle315']
peg_name=['ur16_pegbase','ur16_peg']
qp_ur3=qp_opt(mj_sim.sim,F_des=F_des,optimize=False,hybrid=False,ee_site='ur3:ee',target_site='wpt1',qvel_index=np.array([0,1,2,3,4,5]),peg_name=nozzle_name)


# while ur3_pub.get_num_connections()==0:
#     time.sleep(0.5)

ctrl.movej(mj_sim.init_joints[:6],ur3_pub)

# ctrl.movej(mj_sim.init_joints[6:],ur16_pub)
# commands.append(mj_sim.init_joints)
# time.sleep(2)
# qp_ur3.kp=60*np.ones(6)
# qp_ur3.kd=60*np.ones(6)
qp_ur3.kp=np.array([80,40,60,60,60,60])
qp_ur3.kd=np.array([80,30,60,60,60,60])
s = 20
q_opt=[]
tau_0=[]
countj=0



print('starting simulation')
while mj_sim.sim.data.time<s:
    
    if abs(sensor_array[2]) > 8:
        print('force too high, exiting')
        break
    else:
    # ## Run the QP optimizer ##
        q_out=qp_ur3.run_opt(sensor_array)
        
        ## Step Mujoco Forward ##
        mj_sim.sim.step()
        try:
            (trans,rot) = listener.lookupTransform('base_link', 'tool0', rospy.Time(0))
            countj = countj + 1
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        q_opt.append([pos for pos in mj_sim.sim.data.qvel[:6]])
        tau_0.append([pos for pos in mj_sim.sim.data.sensordata[:3]])
        # a=deepcopy(mj_sim.sim.data.qpos[:6])
        stater.append([pos for pos in mj_sim.sim.data.qpos[:6]])
        position.append([pos for pos in mj_sim.sim.data.geom_xpos[-1]])
        position_ur.append([trans])
        # if countj == 2:
        

        ctrl.servoj(mj_sim.sim.data.qpos[:6],ur3_pub)
        

        #     countj=0
        # else:
        #     countj+=1
        # ctrl.servoj(mj_sim.sim.data.qpos[6:],pub_ur16)
        r.sleep()
        if view==True:
            viewer.render()
print('simulation complete')
t0=np.array(tau_0)
q0=np.array(q_opt)*180/math.pi
# np.savez('./commands.npz',commands=commands)
fig1 = plt.figure()
plt.xlabel('Times(s) normalized')
plt.ylabel('Joint Positions (radians)')
poses = []
timer_array = []
for timer,i in zip(ctrl.timestam,ctrl.statefull):
    poses.append(i)
    timer_array.append(timer-ctrl.timestam[0])
poses = np.array(poses)



# fig2 = plt.figure()
timed = np.linspace(0,s,countj)
stater = np.array(stater)
plt.plot(timed, stater[:,0], label='mujoco', color='r')
plt.plot(timed, stater[:,1], label='mujoco', color='g')
plt.plot(timed, stater[:,2], label='mujoco', color='b')
plt.plot(timed, stater[:,3], label='mujoco', color='c')
plt.plot(timed, stater[:,4], label='mujoco', color='m')
plt.plot(timed, stater[:,5], label='mujoco', color='y')
legend1 = plt.legend(['j1','j2','j3','j4','j5','j6'], loc = 'upper right')
plt.gca().add_artist(legend1)

plt.plot(timer_array, poses[:,0], '--', label='ur arm',color='r')
plt.plot(timer_array, poses[:,1], '--', label='ur arm',color='g')
plt.plot(timer_array, poses[:,2], '--', label='ur arm',color='b')
plt.plot(timer_array, poses[:,3], '--', label='ur arm',color='c')
plt.plot(timer_array, poses[:,4], '--', label='ur arm',color='m')
plt.plot(timer_array, poses[:,5], '--', label='ur arm',color='y')
# y=butter_lowpass_filter(t0[:,2],10,1/mj_sim.sim.model.opt.timestep,5)
position = np.array(position)
fig2 = plt.figure()
plt.plot(timed,(position[:,0]-position[0,0]),color='r')
plt.plot(timed,(position[:,1]-position[0,1]),color='g')
plt.plot(timed,(position[:,2]-position[0,2]),color='b')
position_ur = np.array(position_ur).reshape(len(position_ur),3)
plt.plot(timed,(position_ur[:,0]-position_ur[0,0]),'--', color='r')
plt.plot(timed,(position_ur[:,1]-position_ur[0,1]),'--', color='g')
plt.plot(timed,(position_ur[:,2]-position_ur[0,2]),'--', color='b')

custom_lines = [Line2D([0],[0],color='0', label='mujoco'),
                Line2D([0],[0],color='0',linestyle='--',label='ur_arm')]
plt.legend(handles=custom_lines, loc='lower right')
plt.show()
import numpy as np
import rospy
from ros_utils import ur_controllers
from std_msgs.msg import String

ctrl = ur_controllers()
rospy.init_node('buffer_runner')
pub = rospy.Publisher("/left/ur_hardware_interface/script_command", String, queue_size=1)
npz = np.load('/home/yashas/Documents/cmu/impedance_controller/scripts/commands.npz',allow_pickle=True)
print(npz)
rate = rospy.Rate(10)

for i in range(len(npz['commands'])-20):
    ctrl.servoj(npz['commands'][i],pub)
    print(i)
    rate.sleep()
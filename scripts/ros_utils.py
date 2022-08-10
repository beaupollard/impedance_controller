#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np

class ur_controllers():
    def __init__(self):
        self.prev_msgs=[]
        self.curr_msgs=[]
        self.moveon=False
        self.new_msg=String()

    def movej(self,q,pub,a=1.4,v=1.05,t=0,r =0):
        new_msg = String()
        cmd_str="movej(["
        for i in q:
            cmd_str=cmd_str+str(i)+","

        cmd_str=cmd_str[:-1]+"],a="+str(a)+", v="+str(v)+", t="+str(t)+", r="+str(r)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def movel(self,q,pub,a=1.2,v=0.25,t=0,r =0):
        new_msg = String()
        cmd_str="movel(["
        for i in q:
            cmd_str=cmd_str+str(i)+","

        cmd_str=cmd_str[:-1]+"],a="+str(a)+", v="+str(v)+", t="+str(t)+", r="+str(r)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def servoj(self,q,pub,t=0.008,lookahead=0.1,gain=300):
        new_msg = String()
        cmd_str="servoj(["
        for i in q:
            cmd_str=cmd_str+str(i)+","

        cmd_str=cmd_str[:-1]+"], 0, 0,"+str(t)+", "+str(lookahead)+", "+str(gain)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def jstate_in(self,data):
        jin=np.array(data.position)
        self.jstate=np.array([jin[2],jin[1],jin[0],jin[3],jin[4],jin[5]])

    # def msgs_published(self,data):
    #     if data.data==self.new_msg.data:
    #         self.moveon=True
    #     else:
    #         self.moveon=False
        # self.curr_msgs=data.data
        # if self.curr_msgs==self.prev_msgs:
        #     self.moveon=False
        # else:
        #     self.prev_msgs=data.data
        #     self.moveon=True
        # jin=np.array(data.position)
        # self.jstate=np.array([jin[2],jin[1],jin[0],jin[3],jin[4],jin[5]])


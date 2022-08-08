import numpy as np
import mujoco_py
from scipy.spatial.transform import Rotation as R
import cvxpy as cp
import math

class qp_opt():
    def __init__(self,sim):
        self.sim=sim
        self.ee_site='mep:ee'
        self.vdot = cp.Variable(len(self.sim.data.qacc[:6]))
        self.pos_lower_lims=self.sim.model.jnt_range[:6,0]
        self.pos_upper_lims=self.sim.model.jnt_range[:6,1]
        self.vel_lower_lims=-math.pi*np.ones(6)
        self.vel_upper_lims=math.pi*np.ones(6)
        Jin=np.vstack((np.reshape(self.sim.data.get_site_jacp(self.ee_site)[:],(3,-1)),np.reshape(self.sim.data.get_site_jacr(self.ee_site)[:],(3,-1))))
        self.J_prev=Jin[:6,:6]
        self.dt=sim.model.opt.timestep
    
    def run_opt(self,qacc):


        ## Get Jacobian of end effector ##
        Jin=np.vstack((np.reshape(self.sim.data.get_site_jacp(self.ee_site)[:],(3,-1)),np.reshape(self.sim.data.get_site_jacr(self.ee_site)[:],(3,-1))))
        J=Jin[:6,:6]

        ## Get bias terms (coriollis/gravity) ##
        h=self.sim.data.qfrc_bias[:6]

        ## Get full mass matrix ##
        rc=len(self.sim.data.qvel)
        mm = np.ndarray(shape=(rc ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mm, self.sim.data.qM)
        mm=np.reshape(mm, (rc, rc))
        mass_matrix=mm[:6,:6]

        dJdt=(J-self.J_prev)/self.dt

        self.J_prev=np.array(J)

        qdot_prev=self.sim.data.qvel[:6]
        q_prev=self.sim.data.qpos[:6]
        
        # Construct the problem.
        constraints=[]
        vdot = cp.Variable(len(self.sim.data.qacc[:6]))
        pos_lower_lims=self.sim.model.jnt_range[:6,0]
        pos_upper_lims=self.sim.model.jnt_range[:6,1]
        vel_lower_lims=-math.pi*np.ones(6)
        vel_upper_lims=math.pi*np.ones(6)
        objective = cp.Minimize(cp.sum_squares(vdot-self.sim.data.qacc[:6]))

        constraints.append(vel_lower_lims <= vdot*self.dt+qdot_prev)
        constraints.append(vel_upper_lims >= vdot*self.dt+qdot_prev)
        constraints.append(pos_lower_lims <= vdot*self.dt**2+qdot_prev*self.dt+q_prev)
        constraints.append(pos_upper_lims >= vdot*self.dt**2+qdot_prev*self.dt+q_prev)        
        prob = cp.Problem(objective, constraints)

        result = prob.solve()


        ## Set the calculated torques
        for i in range(6):
            self.sim.data.qacc[i]=vdot.value[i]      

        mujoco_py.cymj._mj_inverse(self.sim.model,self.sim.data)
        for i in range(6):
            self.sim.data.ctrl[i]=self.sim.data.qfrc_inverse[i]
        return vdot.value


from turtle import pos
import numpy as np
import mujoco_py
from scipy.spatial.transform import Rotation as R
import cvxpy as cp
import math

class qp_opt():
    def __init__(self,sim):
        self.sim=sim
        self.ee_site='mep:ee'
        self.target_site='wpt1'
        self.pegmass=self.sim.model.body_mass[self.sim.model.body_name2id('peg')]
        self.acc_index=self.sim.model.sensor_adr[self.sim.model.sensor_name2id('acc')]
        self.force_index=self.sim.model.sensor_adr[self.sim.model.sensor_name2id('force_sensor')]
        self.torque_index=self.sim.model.sensor_adr[self.sim.model.sensor_name2id('torque_sensor')]
        self.qvel_index=np.array([0,1,2,3,4,5])
        self.vdot = cp.Variable(len(self.sim.data.qacc[:6]))
        self.pos_lower_lims=self.sim.model.jnt_range[:6,0]
        self.pos_upper_lims=self.sim.model.jnt_range[:6,1]
        self.vel_lower_lims=-math.pi*np.ones(6)
        self.vel_upper_lims=math.pi*np.ones(6)
        Jin=np.vstack((np.reshape(self.sim.data.get_site_jacp(self.ee_site)[:],(3,-1)),np.reshape(self.sim.data.get_site_jacr(self.ee_site)[:],(3,-1))))
        self.J_prev=Jin[:6,:6]
        self.dt=sim.model.opt.timestep
        self.kp=np.array([120,120,120,120,120,120])
        self.kd=np.array([60,60,60,60,60,60])

    
    def run_opt(self):


        ## Get Jacobian of end effector ##
        J_ori=np.array(self.sim.data.get_site_jacr(self.ee_site).reshape((3, -1))[:, self.qvel_index])     
        J_pos=np.array(self.sim.data.get_site_jacp(self.ee_site).reshape((3, -1))[:, self.qvel_index])  
        J=np.array(np.vstack([J_pos, J_ori]))     
        Jtinv = np.linalg.pinv(np.transpose(J))
        Jinv = np.linalg.pinv(J)

        ## Get bias terms (coriollis/gravity) ##
        fbias=self.sim.data.qfrc_bias[:6]

        ## Get full mass matrix ##
        rc=len(self.sim.data.qvel)
        mm = np.ndarray(shape=(rc ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mm, self.sim.data.qM)
        mm=np.reshape(mm, (rc, rc))
        mass_matrix=mm[:6,:6]

        dJdt=(J-self.J_prev)/self.dt

        self.J_prev=np.array(J)

        ## Me = J^-T M J^-1
        lambda_ori=np.linalg.pinv(J_ori@np.linalg.inv(mass_matrix)@J_ori.T)
        lambda_pos=np.linalg.pinv(J_pos@np.linalg.inv(mass_matrix)@J_pos.T)        
        Me = Jtinv@mass_matrix@Jinv

        ## Ce ##
        Ce = Jtinv@mass_matrix@Jinv@dJdt@Jinv

        current_pos=self.sim.data.get_site_xpos(self.ee_site)
        desired_pos=self.sim.data.get_site_xpos(self.target_site)
        current_velp=self.sim.data.get_site_xvelp(self.ee_site)

        current_ori=self.sim.data.get_site_xmat(self.ee_site)
        desired_ori=self.sim.data.get_site_xmat(self.target_site)
        current_velr=self.sim.data.get_site_xvelr(self.ee_site)
        
        q_err=self.orientation_err(desired_ori,current_ori)

        position_err=np.concatenate((desired_pos-current_pos,q_err))
        velocity_err=-np.concatenate((current_velp,current_velr))

        Xdes=np.diag(self.kp)@position_err+np.diag(self.kd)@velocity_err

        desired_wrench=np.concatenate((lambda_pos@Xdes[:3],lambda_ori@Xdes[3:]))
        tau=np.transpose(J)@(desired_wrench)+fbias

        ## QP Optimizer ##
        qdot_prev=self.sim.data.qvel[:6]
        q_prev=self.sim.data.qpos[:6]

        # Construct the problem.
        constraints=[]
        vdot = cp.Variable(len(self.sim.data.qacc[:6]))
        u = cp.Variable(len(self.sim.data.qacc[:6]))
        pos_lower_lims=self.sim.model.jnt_range[:6,0]
        pos_upper_lims=self.sim.model.jnt_range[:6,1]
        vel_lower_lims=-math.pi/4*np.ones(6)
        vel_upper_lims=math.pi/4*np.ones(6)

        objective = cp.Minimize(cp.sum_squares(u-tau))
        constraints.append(mass_matrix@vdot+fbias==u+self.endeffector_force())
        constraints.append(vel_lower_lims <= vdot*self.dt+qdot_prev)
        constraints.append(vel_lower_lims <= vdot*self.dt+qdot_prev)
        constraints.append(vel_upper_lims >= vdot*self.dt+qdot_prev)
        constraints.append(pos_lower_lims <= vdot*self.dt**2+qdot_prev*self.dt+q_prev)
        constraints.append(pos_upper_lims >= vdot*self.dt**2+qdot_prev*self.dt+q_prev)        
        prob = cp.Problem(objective, constraints)

        result = prob.solve(verbose=True)

        for i in range(6):
            self.sim.data.ctrl[i]=u.value[i]
        return vdot.value

    def orientation_err(self,desired,current):
        rc1 = current[0:3, 0]
        rc2 = current[0:3, 1]
        rc3 = current[0:3, 2]
        rd1 = desired[0:3, 0]
        rd2 = desired[0:3, 1]
        rd3 = desired[0:3, 2]

        error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

        return error

    def endeffector_force(self):
        bias=self.pegmass*self.sim.data.sensordata[self.acc_index:self.acc_index+3]
        Fx=self.sim.data.sensordata[self.force_index:self.force_index+3]-bias
        Tx=self.sim.data.sensordata[self.torque_index:self.torque_index+3]
        R=self.sim.data.get_site_xmat(self.ee_site)
        Fext=-np.concatenate((np.linalg.inv(R)@Fx,np.linalg.inv(R)@Tx))
        return Fext
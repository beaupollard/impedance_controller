from turtle import pos
import numpy as np
import mujoco_py
from scipy.spatial.transform import Rotation as R
import cvxpy as cp
import math

class qp_opt():
    def __init__(self,sim,F_des=np.zeros(6),optimize=True,hybrid=False,ee_site='mep:ee',target_site='wpt1',qvel_index=np.array([0,1,2,3,4,5]),peg_name=['peg','peg_base'],kp=np.array([100,120,120,120,120,120]),kd=np.array([40,60,60,60,60,60])):
        self.sim=sim
        self.ee_site=ee_site
        self.target_site=target_site
        self.pegmass=0
        for i in peg_name:
            self.pegmass+=self.sim.model.body_mass[self.sim.model.body_name2id(i)]
        self.acc_index=self.sim.model.sensor_adr[self.sim.model.sensor_name2id('acc')]
        self.force_index=self.sim.model.sensor_adr[self.sim.model.sensor_name2id('force_sensor')]
        self.torque_index=self.sim.model.sensor_adr[self.sim.model.sensor_name2id('torque_sensor')]
        self.qvel_index=qvel_index
        self.vdot = cp.Variable(len(self.sim.data.qacc[:6]))
        self.pos_lower_lims=self.sim.model.jnt_range[qvel_index,0]
        self.pos_upper_lims=self.sim.model.jnt_range[qvel_index,1]
        self.vel_lower_lims=-math.pi*np.ones(6)
        self.vel_upper_lims=math.pi*np.ones(6)
        J_ori=np.array(self.sim.data.get_site_jacr(self.ee_site).reshape((3, -1))[:, self.qvel_index])     
        J_pos=np.array(self.sim.data.get_site_jacp(self.ee_site).reshape((3, -1))[:, self.qvel_index])          
        self.J_prev=np.array(np.vstack([J_pos, J_ori])) 
        self.dt=sim.model.opt.timestep
        self.kp=kp
        self.kd=kd
        self.contact_count=0
        self.F_des=F_des
        self.optimize=optimize
        self.hybrid=hybrid
        self.force_error_prev=np.zeros(6)
        self.kp_err=1.5*np.ones(6)
        self.kd_err=1*np.ones(6)
        self.ki_err=0.0*np.ones(6)
  
    def run_opt(self):

        ## Get Jacobian of end effector ##
        J_ori=np.array(self.sim.data.get_site_jacr(self.ee_site).reshape((3, -1))[:, self.qvel_index])     
        J_pos=np.array(self.sim.data.get_site_jacp(self.ee_site).reshape((3, -1))[:, self.qvel_index])  
        self.J=np.array(np.vstack([J_pos, J_ori]))     
        Jtinv = np.linalg.pinv(np.transpose(self.J))
        Jinv = np.linalg.pinv(self.J)

        ## Get bias terms (coriollis/gravity) ##
        self.fbias=self.sim.data.qfrc_bias[self.qvel_index]

        ## Get full mass matrix ##
        rc=len(self.sim.data.qvel)
        mm = np.ndarray(shape=(rc ** 2,), dtype=np.float64, order='C')
        # mujoco_py.cymj._mj_fullM(self.sim.model, mm, self.sim.data.qM)
        mujoco_py.functions.mj_fullM(self.sim.model, mm, self.sim.data.qM)
        mm=np.reshape(mm, (rc, rc))
        mass_matrix=mm[self.qvel_index[0]:self.qvel_index[-1]+1,self.qvel_index[0]:self.qvel_index[-1]+1]

        dJdt=(self.J-self.J_prev)/self.dt

        self.J_prev=np.array(self.J)

        ## Me = J^-T M J^-1
        self.lambda_ori=np.linalg.pinv(J_ori@np.linalg.inv(mass_matrix)@J_ori.T)
        self.lambda_pos=np.linalg.pinv(J_pos@np.linalg.inv(mass_matrix)@J_pos.T)        
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

        self.Xdes=np.diag(self.kp)@position_err+np.diag(self.kd)@velocity_err

        tau = self.force_control(hybrid=self.hybrid,F_des=self.F_des)
        for i in range(len(self.qvel_index)):
            self.sim.data.ctrl[self.qvel_index[i]]=tau[i]

        ## Run Convex QP Optimizer to satisfy constraints ## 
        if self.optimize==True:
            self.QP_opt(tau,mass_matrix)

        return self.endeffector_force()

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
        Tx=0*self.sim.data.sensordata[self.torque_index:self.torque_index+3]
        R=self.sim.data.get_site_xmat(self.ee_site)
        Fext=-np.concatenate((np.linalg.inv(R)@Fx,np.linalg.inv(R)@Tx))
        return Fext

    def force_control(self,hybrid=False,F_des=np.zeros(6)):
        buffer=50
        F_ext=self.endeffector_force()

        desired_wrench=np.concatenate((self.lambda_pos@self.Xdes[:3],self.lambda_ori@self.Xdes[3:]))   
        tau_pos=np.transpose(self.J)@(desired_wrench)+self.fbias

        if np.linalg.norm(F_ext)<10**(-10):
            hybrid=False
            self.contact_count=0
            self.int_err=np.zeros(6)
        else:
            self.contact_count+=1

        if hybrid==True:          
            ## Get into end effector frame ##
            sigma=np.zeros(6)
            sigma[np.where(F_des==0)]=1
            # sigma=np.ones(6)
            sigma_bar=np.ones(6)-sigma
            Sf=self.sim.data.get_site_xmat(self.ee_site)
            # omega_f=Sf.T@np.diag(sigma[:3])@Sf
            # omega_t=Sf.T@np.diag(sigma[3:])@Sf         
            omega_f=Sf@np.diag(sigma[:3])@Sf.T
            omega_t=Sf@np.diag(sigma[3:])@Sf.T

            desired_wrench_pos=np.concatenate((self.lambda_pos@omega_f@self.Xdes[:3],self.lambda_ori@omega_t@self.Xdes[3:]))
            # omegabar_f=Sf.T@np.diag(sigma_bar[:3])
            # omegabar_t=Sf.T@np.diag(sigma_bar[3:])
            omegabar_f=Sf@np.diag(sigma_bar[:3])
            omegabar_t=Sf@np.diag(sigma_bar[3:])
            # desired_wrench_force=np.concatenate((self.lambda_pos@omegabar_f@F_des[:3],self.lambda_ori@omegabar_t@F_des[3:]))
            desired_wrench_force=np.concatenate((omegabar_f@F_des[:3],omegabar_t@F_des[3:]))
            # force_err=0*np.concatenate((Sf.T@(-Sf@F_ext[:3]-F_des[:3]),np.zeros(3)))#0*(F_ext-np.concatenate((Sf.T@F_des[:3]@Sf,Sf.T@F_des[3:6]@Sf)))
            force_err=3.0*np.concatenate((Sf@(np.diag(sigma_bar[:3])@(Sf.T@F_ext[:3]-F_des[:3])),np.zeros(3)))
            dforce_errdt=0.1*(force_err-self.force_error_prev)
            err_des=force_err#+dforce_errdt#np.diag(self.kp_err)@force_err+np.diag(self.kd_err)@dforce_errdt#+np.diag(self.ki_err)@self.int_err
            # if self.sim.data.time>8:
            #     err_des=1*force_err
            # if self.contact_count>buffer:
            #     desired_wrench_pos[0]=0
            self.force_error_prev=force_err
            tau_force=np.transpose(self.J)@(desired_wrench_pos+desired_wrench_force-err_des)+self.fbias
            if self.contact_count<buffer:
                tau=(1-self.contact_count/buffer)*tau_pos+self.contact_count/buffer*(tau_force)
            else:
                # inp=desired_wrench_pos+desired_wrench_force-Force_err
                # inp[0]=-2
                # tau=np.transpose(self.J)@inp+self.fbias
                tau=tau_force
                # self.int_err=self.int_err+err_des
        else:
            tau=np.transpose(self.J)@(desired_wrench)+self.fbias

        
        return tau

    def QP_opt(self,tau,mass_matrix):
        ## QP Optimizer ##
        qdot_prev=self.sim.data.qvel[self.qvel_index]
        q_prev=self.sim.data.qpos[self.qvel_index]

        # Construct the problem.
        constraints=[]
        vdot = cp.Variable(len(self.sim.data.qacc[self.qvel_index]))
        u = cp.Variable(len(self.sim.data.qacc[self.qvel_index]))
        pos_lower_lims=self.sim.model.jnt_range[self.qvel_index,0]
        pos_upper_lims=self.sim.model.jnt_range[self.qvel_index,1]
        vel_lower_lims=-math.pi/4*np.ones(6)
        vel_upper_lims=math.pi/4*np.ones(6)

        objective = cp.Minimize(cp.sum_squares(u-tau))
        constraints.append(mass_matrix@vdot+self.fbias==u+self.endeffector_force())
        constraints.append(vel_lower_lims <= vdot*self.dt+qdot_prev)
        constraints.append(vel_lower_lims <= vdot*self.dt+qdot_prev)
        constraints.append(vel_upper_lims >= vdot*self.dt+qdot_prev)
        constraints.append(pos_lower_lims <= vdot*self.dt**2+qdot_prev*self.dt+q_prev)
        constraints.append(pos_upper_lims >= vdot*self.dt**2+qdot_prev*self.dt+q_prev)        
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(solve='MOSEK', verbose=False)

            for i in range(6):
                self.sim.data.ctrl[self.qvel_index[i]]=u.value[i]
        except:
            pass
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm,expm
from icecream import ic
from utils import R3_so3, so3_R3
import os.path as osp
import json
import scipy.io
import math

class FixedWing():
    def __init__(self,config_path=osp.join(osp.dirname(__file__), "pyfly_config.json"),
                 parameter_path=osp.join(osp.dirname(__file__), "x8_param.mat"),
                 config_kw=None):
        _, parameter_extension = osp.splitext(parameter_path)
        if parameter_extension == ".mat":
            self.params = scipy.io.loadmat(parameter_path, squeeze_me=True)
        elif parameter_extension == ".json":
            with open(parameter_path) as param_file:
                self.params = json.load(param_file)
        else:
            raise Exception("Unsupported parameter file extension.")
        
        with open(config_path) as config_file:
            self.cfg = json.load(config_file)
        self.params["ar"] = self.params["b"] ** 2 / self.params["S_wing"]
        self.mb= self.params["mass"]
        self.rho = self.cfg["rho"]
        self.g = 9.81
        self.h = 0.005
        self.dt = 0.01
        self.x = np.zeros(9)
        self.R = np.eye(3)
        self.J = np.array([[self.params["Jx"], 0, -self.params["Jxz"]],
                    [0, self.params["Jy"], 0, ],
                    [-self.params["Jxz"], 0, self.params["Jz"]]
                    ])
        self.f_min = None
        self.tau_min = None
        self.f_max = None
        self.tau_max = None

        for variable in self.cfg["variables"]:
            if variable["name"] == "elevator":
                self.elevator_min = variable["value_min"] 
                self.elevator_max = variable["value_max"]
            elif variable["name"] == "aileron":
                self.aileron_min = variable["value_min"]
                self.aileron_max = variable["value_max"]
                self.rudder_min = variable["value_min"]
                self.rudder_max = variable["value_max"]
            elif variable["name"]== "throttle":
                self.throttle_min = variable["value_min"]
                self.throttle_max = variable["value_max"]
            elif variable["name"] == "omega_q":
                self.omega_q_min = np.deg2rad(variable["constraint_min"])
                self.omega_q_max = np.deg2rad(variable["constraint_max"])
            elif variable["name"] == "omega_r":
                self.omega_r_min = np.deg2rad(variable["constraint_min"])
                self.omega_r_max = np.deg2rad(variable["constraint_max"])
            elif variable["name"] == "omega_p":
                self.omega_p_min = np.deg2rad(variable["constraint_min"])
                self.omega_p_max = np.deg2rad(variable["constraint_max"])
            elif variable["name"] == "velocity_u":
                self.velocity_u_min = variable["init_min"]
                self.velocity_u_max = variable["init_max"]
            elif variable["name"] == "velocity_v":
                self.velocity_v_min = variable["init_min"]
                self.velocity_v_max = variable["init_max"]
            elif variable["name"] == "velocity_w":
                self.velocity_w_min = variable["init_min"]
                self.velocity_w_max = variable["init_max"]

    def _v_dot(self,v_b, w_b, f):
        return f/self.mb - np.cross(w_b,v_b)
    
    def _w_dot(self, w_b, tau):
        return np.linalg.inv(self.J)@(tau - np.cross(w_b, self.J@w_b))
    
    # def _R_dot(self, R, w):
    #     return R@R3_so3(w)
    
    def _R_update(self, w):
        # w[0] *= -1
        # w[2] *= -1
        self.R = self.R@expm(R3_so3(w)*self.h)
    
    def _v(self, v, R):
        return R@v

    def _f(self,t,x, inputs):
        w = x[0:3]
        v_b = x[3:6]
        f = inputs[0:3]
        tau = inputs[3:6]
        dv_dt = self._v_dot(v_b, w, f)
        dw_dt = self._w_dot(w, tau)
        dp_dt = self._v(v_b, self.R)
        return np.hstack((dw_dt, dv_dt, dp_dt))

    def _forces(self,R_attitude, omega, vel, controls):
        elevator, aileron, rudder, throttle = controls
       

        p, q, r = omega

        # if self.wind.turbulence:
        #     p_w, q_w, r_w = self.wind.get_turbulence_angular(self.cur_sim_step)
        #     p, q, r = p - p_w, q - q_w, r - r_w

        Va = np.linalg.norm(vel)#, alpha, beta = self._calculate_airspeed_factors(R_attitude, vel)

        alpha = 0 
        beta = 0 

        pre_fac = 0.5 * self.rho * Va ** 2 * self.params["S_wing"]

        fg_b = R_attitude.T@(self.params["mass"] * np.array([0,0,self.g]) )

        C_L_alpha_lin = self.params["C_L_0"] + self.params["C_L_alpha"] * alpha

        # Nonlinear version of lift coefficient with stall
        a_0 = self.params["a_0"]
        M = self.params["M"]
        e = self.params["e"]  # oswald efficiency
        ar = self.params["ar"]
        C_D_p = self.params["C_D_p"]
        C_m_fp = self.params["C_m_fp"]
        C_m_alpha = self.params["C_m_alpha"]
        C_m_0 = self.params["C_m_0"]

        sigma = (1 + np.exp(-M * (alpha - a_0)) + np.exp(M * (alpha + a_0))) / (
                    (1 + np.exp(-M * (alpha - a_0))) * (1 + np.exp(M * (alpha + a_0))))
        C_L_alpha = (1 - sigma) * C_L_alpha_lin + sigma * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))

        f_lift_s = pre_fac * (C_L_alpha + self.params["C_L_q"] * self.params["c"] / (2 * Va) * q + self.params[
            "C_L_delta_e"] * elevator)


        # C_D_alpha = self.params["C_D_0"] + self.params["C_D_alpha1"] * alpha + self.params["C_D_alpha2"] * alpha ** 2
        C_D_alpha = C_D_p + (1 - sigma) * ((self.params["C_L_0"] + self.params["C_L_alpha"] * alpha) ** 2 / (np.pi * e * ar)) + sigma * (2 * np.sign(alpha) * abs(math.pow(np.sin(alpha), 3)))

        C_D_beta = self.params["C_D_beta1"] * beta + self.params["C_D_beta2"] * beta ** 2
        f_drag_s = pre_fac * (
                    C_D_alpha + C_D_beta + self.params["C_D_q"] * self.params["c"] / (2 * Va) * q + self.params[
                "C_D_delta_e"] * elevator ** 2)

        C_m = (1 - sigma) * (C_m_0 + C_m_alpha * alpha) + sigma * (C_m_fp * np.sign(alpha) * np.sin(alpha) ** 2)
        m = pre_fac * self.params["c"] * (C_m + self.params["C_m_q"] * self.params["b"] / (2 * Va) * q + self.params[
            "C_m_delta_e"] * elevator)

        f_y = pre_fac * (
                    self.params["C_Y_0"] + self.params["C_Y_beta"] * beta + self.params["C_Y_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_Y_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_Y_delta_a"] * aileron + self.params["C_Y_delta_r"] * rudder)
        l = pre_fac * self.params["b"] * (
                    self.params["C_l_0"] + self.params["C_l_beta"] * beta + self.params["C_l_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_l_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_l_delta_a"] * aileron + self.params["C_l_delta_r"] * rudder)
        n = pre_fac * self.params["b"] * (
                    self.params["C_n_0"] + self.params["C_n_beta"] * beta + self.params["C_n_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_n_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_n_delta_a"] * aileron + self.params["C_n_delta_r"] * rudder)

        # f_aero = np.dot(self._rot_b_v(np.array([0, alpha, beta])), np.array([-f_drag_s, f_y, -f_lift_s]))
        f_aero =  np.array([-f_drag_s, f_y, -f_lift_s])
        tau_aero = np.array([l, m, n])
        # Vd = Va + throttle * (self.params["k_motor"] - Va)
        # f_prop = np.array([0.5 * self.rho * self.params["S_prop"] * self.params["C_prop"] * Vd * (Vd - Va), 0, 0])
        f_prop = np.array([0.5 * self.rho * self.params["S_prop"] * self.params["C_prop"] *  ((self.params["k_motor"]*throttle)**2 - Va**2), 0, 0])
        tau_prop = np.array([-self.params["k_T_P"] * (self.params["k_Omega"] * throttle) ** 2, 0, 0])
        f = f_prop + fg_b + f_aero
        tau = tau_aero + tau_prop

        return f, tau

    def _forces_min(self, R_attitude):

        vel_min = np.array([self.velocity_u_min, self.velocity_v_min, self.velocity_w_min])
        self.f_min, self.tau_min = self._forces(R_attitude, [self.omega_p_min, self.omega_q_min, self.omega_r_min], vel_min,
                     [self.elevator_min, self.aileron_min, self.rudder_min, self.throttle_min])
        
        return self.f_min, self.tau_min

    def _forces_max(self, R_attitude):

        vel_max = np.array([self.velocity_u_max, self.velocity_v_max, self.velocity_w_max])
        self.f_max, self.tau_max = self._forces(R_attitude, [self.omega_p_max, self.omega_q_max, self.omega_r_max], vel_max,
                     [self.elevator_max, self.aileron_max, self.rudder_max, self.throttle_max])
        return self.f_max, self.tau_max
        
    def _runge_kutta(self, x, inputs):
        N = int(self.dt/self.h)
        t = 0
        for i in range(N):
            t += self.h
            k1 = self.h*self._f(t ,x, inputs)
            k2 = self.h*self._f(t+0.5*self.h ,x + k1/2,inputs)
            k3 = self.h*self._f(t+0.5*self.h ,x + k2/2,inputs)
            k4 = self.h*self._f(t + self.h ,x + k3, inputs)
            w = x[0:3]
            self._R_update(w)
            x = x + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            x[0:3] = np.clip(x[0:3], -np.pi, np.pi)
        return x
    
    def step(self, inputs):
        self.x = self._runge_kutta(self.x, inputs)
        return self.x, self.R


import numpy as np
# from finite_pid_embedding import PID_fixed_wing
# from finite_lqr_embedding import LQR_fixed_wing
from finite_mpc_embedding import MPC_fixed_wing
from embedding_SO3_sim import Embedding
import matplotlib.pyplot as plt
from icecream import ic



r = 190
T_p = 40
phi_dot = 2*np.pi/T_p
ic(phi_dot)
t_max =4
t_min = 0
k_phi = 15
tactic = ''
n_agents = 1
controller = MPC_fixed_wing(t_max, t_min, r, T_p)
Nh = controller.Nh
dt = controller.T
embedding = Embedding(r,phi_dot,k_phi,tactic,n_agents,dt)

N = int((t_max-t_min)/dt)  #number of points

#setting the initial position
initial_position = np.array([r*np.cos(0),r*np.sin(0),2*r]).reshape(-1,1)

_,target_r_new, target_v_new, target_a_new, _, _ = embedding.targets(initial_position, np.array([0,0,0]).reshape(-1,1))
target_r = target_r_new
target_v = target_v_new
target_a = target_a_new
#initializing the controller
_,target_r_new, target_v_new, target_a_new, _, _  = embedding.targets(target_r, target_v)
target_r = target_r_new
target_v = target_v_new
target_a = target_a_new
_,target_r_new, target_v_new, target_a_new, _, _ = embedding.targets(target_r, target_v)
target_r = target_r_new
target_v = target_v_new
target_a = target_a_new
controller.references(0,target_r[:,0], target_v[:,0], target_a[:,0])
controller.set_initial_conditions(0)
_,target_r_new, target_v_new, target_a_new, _, _ = embedding.targets(target_r, target_v)
target_r = target_r_new
target_v = target_v_new
target_a = target_a_new
controller.references(1,target_r[:,0], target_v[:,0], target_a[:,0])
controller.set_initial_conditions(1)
for i in range(1,N-1):

    _,target_r_new, target_v_new, target_a_new, _, _ = embedding.targets(target_r,target_v)
    target_r = target_r_new
    target_v = target_v_new
    target_a = target_a_new
    controller.references(i,target_r[:,0], target_v[:,0], target_a[:,0])
    controller.calc_A_and_B(i)

# for i in range(N,1,-1):
#     controller.calc_K_lqr(i)


pos_real = target_r
vel_real = target_v
# controller.plot_references()
plt.show()
for i in range(1,N-Nh):
    # _,target_r, target_v, target_a, _, _ = embedding.targets(pos_real,vel_real)
    np.clip(target_a, -10, 10)
    np.clip(target_v, -40, 40)
    controller.control(i, target_r[:,0], target_v[:,0], target_a[:,0])
    pos_real = controller.X[6:9,i+1].reshape(-1,1)
    vel_real = controller.X[3:6,i+1].reshape(-1,1)

# controller.plot_angles()
controller.plot_3D()
controller.plot_erros()
# controller.plot_controls()
# controller.plot_velocity_body()
# controller.plot_angles()
controller.plot_force_omega()
controller.plot_velocity()
controller.plot_positions()
# controller.plot_error_position()
# controller.plot_error_velocity()
plt.show()



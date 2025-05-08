import numpy as np
# from finite_pid_embedding import PID_fixed_wing
# from finite_lqr_embedding import LQR_fixed_wing
from finite_mpc_embedding import MPC_fixed_wing
from embedding_SO3_sim import Embedding
import matplotlib.pyplot as plt
from icecream import ic
import random
import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True



r = 190
T_p = 40
phi_dot = 2*np.pi/T_p
ic(phi_dot)
t_max = 5

t_min = 0
k_phi = 1
tactic = 'circle' #circle, spiral, line
n_agents = 1
controllers = []
for i in range(n_agents):
    controllers.append(MPC_fixed_wing(t_max, t_min, r, T_p, str(i+1), noise = False, save = True))
Nh = controllers[0].Nh
dt = controllers[0].T
embedding = Embedding(r,phi_dot,k_phi,tactic,n_agents,dt)

N = int((t_max-t_min)/dt)  #number of points
target_r = np.zeros((3, n_agents))
target_v = np.zeros((3, n_agents))
target_a = np.zeros((3, n_agents))
phi_diff = np.zeros((3,N))
#setting the initial position
for i in range(n_agents):
    offset = 0 #(-1)**i * np.deg2rad(random.uniform(0, 20))
    target_r[:,i] = np.array([r * np.cos(i * 2 * np.pi / n_agents + offset), 
                              r * np.sin(i * 2 * np.pi / n_agents + offset), 
                              2 * r])


for j in range(3):
    _,target_r, target_v, target_a, _, _ = embedding.targets(target_r, target_v, False)

for i in range(n_agents):
    controllers[i].references(0,target_r[:,i], target_v[:,i], target_a[:,i])
    controllers[i].set_initial_conditions(0)

phi,target_r, target_v, target_a, phi_diff_new, _ = embedding.targets(target_r, target_v,False)
phi_diff[:,0] = phi_diff_new

for i in range(n_agents):
    controllers[i].references(1,target_r[:,i], target_v[:,i], target_a[:,i])
    controllers[i].set_initial_conditions(1)
    

pos_real = target_r
vel_real = target_v
# controller.plot_references()
# plt.show()
for i in range(2,N-Nh):
    _,target_r_new, target_v_new, target_a_new, phi_diff_new, _ = embedding.targets(pos_real,vel_real,True)
    for j in range(i,Nh+i):
        _,target_r_new, target_v_new, target_a_new, phi_diff_new, _ = embedding.targets(pos_real,vel_real,True)
        if j == i:
            target_r = target_r_new
            target_v = target_v_new
            target_a = target_a_new
            phi_diff[:,j] = phi_diff_new
        pos_real = target_r_new
        vel_real = target_v_new
        for k in range(n_agents):
            controllers[k].references(j,target_r_new[:,k], target_v_new[:,k], target_a_new[:,k])
            controllers[k].calc_A_and_B(j)
    # np.clip(target_a, -10, 10)
    # np.clip(target_v, -40, 40)

    for k in range(n_agents):
        if i == 2:
            controllers[k].set_initial_conditions(2)
        # ic(controllers[k].va_r_dot_body[:,i], controllers[k].wr_r[:,i],controllers[k].va_r_body[:,i],controllers[k].f_r[:,i],controllers[k].tau_r[:,i], controllers[k].d_wr_r[:,i])
        controllers[k].control(i, target_r[:,k], target_v[:,k], target_a[:,k])
    pos_real = target_r#controllers[k].X[6:9,i].reshape(-1,1)#
    vel_real = controllers[k].X[3:6,i].reshape(-1,1)
    # vel_real = target_v

for i in range(n_agents):
    controllers[i].plot_angles()
    controllers[i].plot_3D()
    controllers[i].plot_erros()
    controllers[i].plot_controls()
    controllers[i].plot_acceleration()
    # controller.plot_velocity_body()
    # controller.plot_angles()
    controllers[i].plot_force_omega()
    controllers[i].plot_velocity()
    # controllers[i].plot_positions()
    # controller.plot_error_position()
    # controller.plot_error_velocity()
fig = plt.figure()
plt.plot(np.rad2deg(phi_diff[0,:-Nh]), label='phi_diff 1-2')
plt.plot(np.rad2deg(phi_diff[1,:-Nh]), label='phi_diff 2-3')
plt.plot(np.rad2deg(phi_diff[2,:-Nh]), label='phi_diff 3-1')
plt.title('phi_diff')
plt.xlabel('time')
plt.ylabel('phi_diff')
plt.legend()
plt.grid()
plt.savefig('phi_diff_fixed_wing.png', dpi=300)
# plt.save(fig, 'phi_diff.png', dpi=300)
plt.show()



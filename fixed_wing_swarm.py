import numpy as np
# from finite_pid_embedding import PID_fixed_wing
# from finite_lqr_embedding import LQR_fixed_wing
from finite_mpc_embedding import MPC_fixed_wing
from embedding_SO3_sim import Embedding
import matplotlib.pyplot as plt
from icecream import ic
import random
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
plt.rcParams.update({
    # 'font.family': 'Times New Roman',
    'font.size': 13
})

figures_path = "/home/dimitria/fixed_wing/figures"
r = 10
T_p = 10
phi_dot = 2*np.pi/T_p
ic(phi_dot)
t_max = 10.1

t_min = 0
k_phi = 0.5
tactic = '' #circle, spiral, line
n_agents = 3
controllers = []
for i in range(n_agents):
    controllers.append(MPC_fixed_wing(t_max, t_min, r, T_p, str(i+1),figures_path, noise = True, save = True))
Nh = controllers[0].Nh
dt = controllers[0].T
embedding = Embedding(r,phi_dot,k_phi,tactic,n_agents,dt)

N = int((t_max-t_min)/dt)  #number of points
target_r = np.zeros((3, n_agents))
target_v = np.zeros((3, n_agents))
target_a = np.zeros((3, n_agents))
phi_diff = np.zeros((n_agents,N))
distances = np.zeros((n_agents,N))
#setting the initial position
for i in range(n_agents):
    offset = i *np.deg2rad(20)# (-1)**i * np.deg2rad(random.uniform(0, 20))
    target_r[:,i] = np.array([r * np.cos(i * 2 * np.pi / n_agents + offset), 
                              r * np.sin(i * 2 * np.pi / n_agents + offset), 
                              2 * r])
    ic(np.rad2deg(i * 2 * np.pi / n_agents + offset))


phi = np.zeros(n_agents)
for j in range(3):

    phi,target_r, target_v,target_a, phi_diff_new, dist = embedding.targets(target_r, target_v, phi, False)
phi_diff[:,0] = phi_diff_new
distances[:,0] = dist

for i in range(n_agents):
    controllers[i].references(0,target_r[:,i], target_v[:,i], target_a[:,i])
    controllers[i].set_initial_conditions(0)

phi,target_r, target_v, target_a, phi_diff_new, dist = embedding.targets(target_r,target_v,phi,True)
phi_diff[:,1] = phi_diff_new
distances[:,1] = dist

for i in range(n_agents):
    controllers[i].references(1,target_r[:,i], target_v[:,i], target_a[:,i])
    controllers[i].set_initial_conditions(1)
    

pos_real = target_r
vel_real = target_v
# controller.plot_references()
# plt.show()
for i in range(2,N-Nh):
    # ic(i)
    # phi,target_r_new, target_v_new, target_a_new, phi_diff_new, _ = embedding.targets(pos_real,vel_real, phi, True)
    for j in range(i,Nh+i):
        phi,target_r_new, target_v_new, target_a_new, phi_diff_new, dist = embedding.targets(pos_real,vel_real, phi, True)
        if j == i:
            target_r = target_r_new
            target_v = target_v_new
            target_a = target_a_new
            phi_diff[:,j] = phi_diff_new
            distances[:,j] = dist
        pos_real = target_r_new
        vel_real = target_v_new
        for k in range(n_agents):
            controllers[k].references(j,target_r_new[:,k], target_v_new[:,k], target_a_new[:,k])
            controllers[k].calc_A_and_B(j)
    # np.clip(target_a, -10, 10)
    # np.clip(target_v, -40, 40)
    for l in range(1):
        for k in range(n_agents):
            if i ==2:
                controllers[k].set_initial_conditions(2)
        # ic(controllers[k].va_r_dot_body[:,i], controllers[k].wr_r[:,i],controllers[k].va_r_body[:,i],controllers[k].f_r[:,i],controllers[k].tau_r[:,i], controllers[k].d_wr_r[:,i])
            controllers[k].control(i, target_r[:,k], target_v[:,k], target_a[:,k])
            pos_real[:,k] = controllers[k].X[6:9,i]#.reshape(-1,n_agents)#target_r#
            vel_real[:,k] = controllers[k].X[3:6,i]#.reshape(-1,n_agents)
    # vel_real = target_v

for i in range(n_agents):
    controllers[i].plot_angles()
    controllers[i].plot_3D()
    controllers[i].plot_erros()
    # controllers[i].plot_controls()
    # controllers[i].plot_acceleration()
    # controller.plot_velocity_body()
    # controller.plot_angles()
    controllers[i].plot_force_omega()
    # controllers[i].plot_velocity()
    controllers[i].plot_positions()
    # controller.plot_error_position()
    # controller.plot_error_velocity()
fig = plt.figure(constrained_layout=True)
plt.plot(np.rad2deg(phi_diff[0,:-Nh]), label='1-2',color='red')
plt.plot(np.rad2deg(phi_diff[1,:-Nh]), label='1-3',color='blue')
plt.plot(np.rad2deg(phi_diff[2,:-Nh]), label='2-3',color='black')
plt.title('Angular Separation Between the Agents ')
plt.xlabel('t (s)')
plt.ylabel('$\phi_{diff}$ (degrees)')
plt.legend()
plt.grid()
# fig.tight_layout()
plt.savefig('phi_diff_fixed_wing.png', dpi=300, bbox_inches='tight')


fig = plt.figure(constrained_layout=True)
plt.plot(distances[0,:-Nh], label='1-2',color='red')
plt.plot(distances[1,:-Nh], label='1-3',color='blue')
plt.plot(distances[2,:-Nh], label='2-3',color='black')
plt.title("Distance Between the Agents")
plt.xlabel('t (s)')
plt.ylabel('Distance (m)')
plt.legend()
plt.grid()
# fig.tight_layout()
plt.subplots_adjust(left=0.2)
plt.savefig(f"{figures_path}/distances_fixed_wing.png", dpi=300, bbox_inches='tight')
# plt.save(fig, 'phi_diff.png', dpi=300)
plt.show()



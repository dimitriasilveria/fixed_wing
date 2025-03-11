import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import R3_so3, so3_R3
from icecream import ic
from scipy.linalg import expm, logm
def references(fixed_wing, x_dot, x_dot_dot, dt,N,Car,n_agents,Wr_r):
    for i in range(N-1):

        for a in range(n_agents):
            X = np.arctan2(x_dot[1,i,a],x_dot[0,i,a])

            ca_1 = np.array([np.cos(X),np.sin(X),0]).T #auxiliar vector 
            attitude = R.from_matrix(Car[:,:,i,a]).as_quat()
            

            fa_r = fixed_wing.simplified_forces(attitude, Car[:,:,i,a], x_dot[:,i,a], x_dot_dot[:,i,a],Wr_r[:,a])
            r1 = x_dot[:,i,a].reshape(3,1)/np.linalg.norm(x_dot[:,i,a])
            if np.linalg.norm(fa_r) != 0:
                r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
            else:
                r3 = np.zeros((3,1))

            aux = R3_so3(r3)@r1
            if np.linalg.norm(aux) != 0:
                r2 = aux.reshape(3,1)/np.linalg.norm(aux)
            else:
                r2 = np.zeros((3,1))

            Car[:,:,i+1,a] = np.hstack((r1, r2, r3))

            if np.linalg.det(Car[:,:,i,a]) != 0 and np.linalg.det(Car[:,:,i+1,a]) != 0:

                Wr_r[:,a] = so3_R3(logm(np.linalg.inv(Car[:,:,i,a])@Car[:,:,i+1,a]))/dt
            else:
                Wr_r[:,a] = np.zeros(3)
            yaw = R.from_matrix(Car[:,:,i,a]).as_euler('zyx')[0]
    return Car, Wr_r
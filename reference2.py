import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import R3_so3, so3_R3
from icecream import ic
from scipy.linalg import expm, logm
def references(fixed_wing, va_r, va_r_dot, dt,N,Car,n_agents,Wr_r):
    z_w = np.array([0,0,1])
    for i in range(N-1):


        for a in range(n_agents):

            #attitude = R.from_matrix(Car[:,:,i,a]).as_quat()
            norm_x_B = np.linalg.norm(va_r[:,a,i])
            norm_aux = np.linalg.norm(va_r_dot[:,a,i]-(va_r_dot[:,a,i]@va_r[:,a,i])*va_r[:,a,i]) 
            if (norm_x_B != 0) and (norm_aux != 0):
                x_B = va_r[:,a,i]/norm_x_B
                R = norm_x_B**2/norm_aux
                phi = np.arctan2(norm_x_B**2,(g*R))
                z_intermediate = z_w - (z_w@x_B)*x_B
                norm_z_intermediate = np.linalg.norm(z_intermediate)
                if norm_z_intermediate != 0:
                    z_intermediate = z_intermediate/norm_z_intermediate
                    z_B = np.cos(phi)*z_intermediate + np.sin(phi)*(np.cross(x_B,z_intermediate))
                    z_B = z_B/np.linalg.norm(z_B)
                    y_B = np.cross(z_B,x_B)
                    y_B = y_B/np.linalg.norm(y_B)
                else:
                    y_B = np.zeros(3)
                    z_B = np.zeros(3)
            else:
                x_B = np.zeros(3)
                z_B = np.zeros(3)
                y_B = np.zeros(3)
                

            Car[:,:,i,a] = np.hstack((x_B.reshape(3,1), y_B.reshape(3,1), z_B.reshape(3,1)))
            if np.linalg.det(Car[:,:,i,a]) != 0 and np.linalg.det(Car[:,:,i+1,a]) != 0:

                Wr_r[:,a] = so3_R3(logm(np.linalg.inv(Car[:,:,i,a])@Car[:,:,i+1,a]))/dt
            else:
                Wr_r[:,a] = np.zeros(3)
    return Car, Wr_r
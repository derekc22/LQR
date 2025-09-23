import numpy as np
import mujoco


global A, Q, B, R

q1 = 1
q2 = 1
r1 = 1


Q = np.array([
    [q1**2, 0],
    [0, q2**2]
])

R = np.array([r1])


def lqr(x):
    
    u = - np.inverse(R) @ B.T @ P @ x
    
    

def main():
    
    m = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
    d = mujoco.MjData(m)

    viewer = mujoco.viewer.launch_passive(m, d)
    
    t = 0
    while t < 1000000000:
        mujoco.mj_step(m, d)
        viewer.sync()
        t += 1

    
    



if __name__ == "__main__":
    main()
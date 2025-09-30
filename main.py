import numpy as np
import mujoco
from sympy import symbols, Matrix, solve, zeros, pprint
from utils import *
from derivation import Aup, Bup, mc, mp, l, Icm, g
from scipy.linalg import solve_continuous_are


# Scalars
p = symbols( "p11 p12 p13 p14 p22 p23 p24 p33 p34 p44")
p11, p12, p13, p14, p22, p23, p24, p33, p34, p44 = p


q11 = 30       # angle (very important)
q12 = 0
q13 = 0
q14 = 0
q22 = 25       # angular velocity (secondary)
q23 = 0
q24 = 0
q33 = 20       # cart position (less important)
q34 = 0
q44 = 5        # cart velocity (least important)

r1 = 0.1

# Define cost matrices
Q = np.array([
    [q11, q12, q13, q14],
    [q12, q22, q23, q24],
    [q13, q23, q33, q34],
    [q14, q24, q34, q44],
])

R = np.array([[r1]])


P = Matrix([
    [p11, p12, p13, p14],
    [p12, p22, p23, p24],
    [p13, p23, p33, p34],
    [p14, p24, p34, p44],
])

    

def get_q(d):
    return np.array([
        d.qpos[1], 
        d.qvel[1], 
        d.qpos[0], 
        d.qvel[0]
    ])




def main():
    

    m, d = load_model("assets/inverted_pendulum.xml")
    reset(m, d, "up")

    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
    to_val = {
        mc: get_body_mass(m, "cart"),
        mp: get_body_mass(m, "pole"),
        l: get_body_size(m, "pole")[1],
        Icm: get_body_inertia(m, "pole")[1], # Iyy
        g: 9.81,
    }


    A = Aup.xreplace(to_val)
    B = Bup.xreplace(to_val)


    # V = P@A + A.T@P + Q - P@B@np.linalg.inv(R)@B.T@P
    # candidates = solve(V - zeros(4), p)
    # sol = [ c for c in candidates if all(v > 0 for v in c) ][0]
    # K_lqr = R @ B.T @ P.subs(list(zip(p, sol)))

    P = solve_continuous_are(np.array(A).astype(float), 
                             np.array(B).astype(float), 
                             Q, R)
    
    K_lqr = np.linalg.inv(R) @ B.T @ P


    
    for t in range(1000000000):

        q = get_q(d)
        u = - K_lqr @ q
        # n = 7*np.random.rand()
        d.ctrl = u #+ n
        mujoco.mj_step(m, d)
        # print(u)
        
        viewer.sync()

    
    



if __name__ == "__main__":
    main()
    # x = Matrix([np.pi, 0])
    # lqr(x)
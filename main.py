import numpy as np
import mujoco
from sympy import symbols, Matrix, solve, zeros, pprint
from utils import *

# Scalars
p1, p2, p3 = symbols("p1 p2 p3")

q11 = 1
q12 = 0
q21 = 0
q22 = 1
r1 = 1
m = 1
l = 1
g = -9.81



# 1) Full nonlinear dynamics:                              ml^2*θ̈ + mglsin(θ) = u
# 2) Isolate θ̈:                                            θ̈ = 1/ml^2 * (u-mglsin(θ))
# 3) Define state variables:                               x1 = θ, x2 = θ̇
# 4) Define state vector x(t) and input vector u(t):       x(t) = [x1; x2], u(t) = [u]
# 5) Form state equation ẋ(t):                             ẋ(t) = [ẋ1; ẋ2] = [x2; 1/ml^2 * (u-mgl*sin(x1))]
# 6) Solve for equilibrium points, x̄:                      ẋ(t) = 0 for u = 0 → x̄ = (nπ, 0)
# 7) Linearize dynamics around x̄ via Taylor Series:        ẋ(t) = Ax(t) + Bu(t)

# Step 7) yields
A = Matrix([
    [0, 1],
    [-g/l, 0]    
])

B = Matrix([0, 1/m*l**2])

# Define cost matrices
Q = Matrix([
    [q11, q12],
    [q21, q22]
])

R = Matrix([r1])

P = Matrix([
    [p1, p2],
    [p2, p3]    
])

V = P@A + A.T@P + Q - P@B@R.inv()@B.T@P

def lqr(x):
    
    candidates = solve(V - zeros(2), (p1, p2, p3))
    
    sol = [ c for c in candidates if all(v > 0 for v in c) ][0]
    # print(sol)
    # exit()

    K_lqr = - R @ B.T @ P.subs([ (p1, sol[0]), (p2, sol[1]), (p3, sol[2]) ])
    u = K_lqr @ x
    
    pprint(u)
    
    

def main():
    
    m, d = load_model("inverted_pendulum.xml")
    reset(m, d, "up")

    viewer = mujoco.viewer.launch_passive(m, d)
    
    for t in range(1000000000):

        d.ctrl = lqr(d.qpos)
        mujoco.mj_step(m, d)
        
        viewer.sync()

    
    



if __name__ == "__main__":
    main()
    # x = Matrix([np.pi, 0])
    # lqr(x)
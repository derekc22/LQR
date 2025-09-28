from sympy import symbols, Function, diff, sin, cos, Derivative, pprint, Eq, solve, Matrix, zeros


def pprint_(expr):
    pprint(expr.xreplace(to_sym))

def pprint_v(expr):
    pprint_(expr.expand())

# constants
mc, mp, l, Icm, g = symbols('mc mp l Icm g')

# variables
t, u = symbols('t u')


theta = Function('θ')(t)
xc = Function('xc')(t)
xp = xc - (l/2)*sin(theta)
yp = (l/2)*cos(theta)


xc_dot = diff(xc, t)
xp_dot = diff(xp, t)
yp_dot = diff(yp, t)
theta_dot = diff(theta, t)


Tcart = (1/2)*mc*xc_dot**2
Tpole = (1/2)*mp*(xp_dot**2 + yp_dot**2) + (1/2)*Icm*theta_dot**2
T = Tcart + Tpole

Vcart = 0
Vpole = mp*g*(l/2)*cos(theta)
V = Vcart + Vpole


# Construct lagrangian
L = T - V

# Construct euler-lagrange EOMs
partialL_theta_sym = diff(L, theta)
partialL_theta_dot_sym = diff(L, theta_dot)
ddt_partialL_theta_dot_sym = diff(partialL_theta_dot_sym, t)

partialL_xc_sym = diff(L, xc)
partialL_xc_dot_sym = diff(L, xc_dot)
ddt_partialL_xc_dot_sym = diff(partialL_xc_dot_sym, t)

eom_theta = Eq(ddt_partialL_theta_dot_sym - partialL_theta_sym, 0)
eom_xc = Eq(ddt_partialL_xc_dot_sym - partialL_xc_sym, u)

# Isolate ẍc, θ̈ 
sol = solve( (eom_theta, eom_xc), (Derivative(xc, t, 2), Derivative(theta, t, 2)) )
xc_ddot = sol[Derivative(xc, (t, 2))]
theta_ddot = sol[Derivative(theta, (t, 2))]

# Define state variables: x1 = θ, x2 = θ̇, x3 = xc, x4 = ẋc
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')

to_state = {
    theta: x1,
    theta_dot: x2,
    xc: x3,
    xc_dot: x4
}

xt = Matrix([
    theta, 
    theta_dot, 
    xc, 
    xc_dot
]).subs(to_state)

xt_dot = Matrix([
    theta_dot, 
    theta_ddot, 
    xc_dot,
    xc_ddot
]).subs(to_state)

ut = Matrix([u])




sol_eq = solve( 
    xt_dot.subs([(u, 0)]) - zeros(4, 1), 
    (x1, x2, x3, x4) 
)


to_up = list(zip((x1, x2, x3, x4), sol_eq[0]))
Aup = xt_dot.jacobian(xt).subs(to_up)
Bup = xt_dot.jacobian(ut).subs(to_up)
# pprint(Aup)
# pprint(Bup)

to_down = list(zip((x1, x2, x3, x4), sol_eq[1]))
Adown = xt_dot.jacobian(xt).subs(to_down)
Bdown = xt_dot.jacobian(ut).subs(to_down)
# pprint(Adown)




# # symbols
# xc_dot_sym, xc_ddot_sym, xp_dot_sym, yp_dot_sym, theta_dot_sym, theta_ddot_sym = symbols(
#     "ẋc ẍc ẋp ẏp θ̇ θ̈"
# )
# to_sym = {
#     xc_dot: xc_dot_sym,
#     xc_ddot: xc_ddot_sym,
#     xp_dot: xp_dot_sym,
#     yp_dot: yp_dot_sym,
#     theta_dot: theta_dot_sym,
#     theta_ddot: theta_ddot_sym,
# }




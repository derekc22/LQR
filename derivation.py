# from sympy import symbols, Matrix, solve, zeros, pprint, sin, cos, diff, expand
# xc, xc_dot, yc, yc_dot, mc, xp, xp_dot, yp, yp_dot, mp, theta, theta_dot, l = symbols("xc ẋc yc ẏc mc xp ẋp yp ẏp mp θ θ̇ l")

# xp = xc - (l/2)*sin(theta)
# xp_dot = xc_dot - (l/2)*cos(theta)*theta_dot
# yp = (l/2)*cos(theta)
# yp_dot = (l/2)*sin(theta)*theta_dot


# Tcart = (1/2)*mc*xc_dot**2
# Tpole = (1/2)*mp*(xp_dot**2 + yp_dot**2)
# T = Tcart + Tpole

# pprint(T)



from sympy import symbols, Function, diff, sin, cos, Derivative, pprint, Eq, solve


def pprint_(expr):
    pprint(expr.xreplace(to_sym))

def pprint_v(expr):
    pprint_(expr.expand())

# constants
mc, mp, l, Icm = symbols('mc mp l Icm')

# variables
t, u = symbols('t u')

# symbols
xc_dot_sym, xc_ddot_sym, xp_dot_sym, yp_dot_sym, theta_dot_sym, theta_ddot_sym = symbols(
    "ẋc ẍc ẋp ẏp θ̇ θ̈"
)

theta = Function('θ')(t)
xc = Function('xc')(t)
yc = Function('yc')(t)
xp = xc - (l/2)*sin(theta)
yp = (l/2)*cos(theta)

to_sym = {
    Derivative(xc, t): xc_dot_sym,
    Derivative(xc, t, 2): xc_ddot_sym,
    Derivative(xp, t): xp_dot_sym,
    Derivative(yp, t): yp_dot_sym,
    Derivative(theta, t): theta_dot_sym,
    Derivative(theta, t, 2): theta_ddot_sym,
}

xc_dot = diff(xc, t)
xp_dot = diff(xp, t)
yp_dot = diff(yp, t)
theta_dot = diff(theta, t)


Tcart = (1/2)*mc*xc_dot**2
Tpole = (1/2)*mp*(xp_dot**2 + yp_dot**2) + (1/2)*Icm*theta_dot**2
T = Tcart + Tpole

Vcart = 0
Vpole = mp*(l/2)*cos(theta)
V = Vcart + Vpole


# construct lagrangian
L = T - V

# construct euler-lagrange EOM
partialL_theta_sym = diff(L, theta)
partialL_theta_dot_sym = diff(L, theta_dot)
ddt_partialL_theta_dot_sym = diff(partialL_theta_dot_sym, t)

partialL_xc_sym = diff(L, xc)
partialL_xc_dot_sym = diff(L, xc_dot)
ddt_partialL_xc_dot_sym = diff(partialL_xc_dot_sym, t)

eom_theta = Eq(ddt_partialL_theta_dot_sym - partialL_theta_sym, 0)
eom_xc = Eq(ddt_partialL_xc_dot_sym - partialL_xc_sym, u)

sol = solve([eom_theta, eom_xc], [Derivative(xc, t, 2), Derivative(theta, t, 2)])
print(sol)



# define x = 

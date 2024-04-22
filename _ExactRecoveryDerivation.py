from sympy import *
rho, epsilon = symbols('r e')
init_printing()
temp0 = (1-2*rho)*(epsilon-1)/log(epsilon)
# print(temp0)
temp1 = temp0 + sqrt(temp0**2+4*rho*(1-rho)*epsilon)
g3 = temp0 * log(temp1 / (2*(1-rho)))-2*rho*(1-rho)*epsilon/temp1-0.5 * temp1
d = g3 + 1 - 2*rho+2*rho*sqrt(epsilon)
print("Difference", latex(d))
derivative = diff(d, rho)
print("Derivative of Difference with rho", latex(derivative))
rangelessthan0 = reduce_inequalities(derivative<0, [rho])
print("Range for Derivative<0 with rho", latex(rangelessthan0))

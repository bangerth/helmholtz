# A little script that helps with evaluating the pressure and velocity at
# arbitrary points along the x-axis.

import cmath

c=343.287
omega=2*3.1415926*100000
rho=1.205728
L=0.004
j = complex(0,1)


def p(x) :
    return (cmath.exp(j*omega/c*x) - cmath.exp(j*omega/c*(2*L-x)) ) / ( 1-cmath.exp(2*j*omega/c*L) )

def u(x) :
    return -1/(rho*c) * (cmath.exp(j*omega/c*x) + cmath.exp(j*omega/c*(2*L-x)) ) / ( 1-cmath.exp(2*j*omega/c*L) )


# The evaluation point chosen in the input file:
x = 0.003
print "Pressure at ", x, ": ", p(x)
print "Velocity at ", x, ": ", u(x)

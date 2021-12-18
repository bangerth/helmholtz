# A little script that helps with evaluating the pressure and velocity at
# arbitrary points along the x-axis.
#
# This is the script where the source is at port=2 and the receiver at
# port=1.

import cmath

# For the case with no attenuation
c=343.287
rho=1.205728

# For the case with attenuation
#c=complex(291.437,47.84)
#rho=complex(1.5845,-0.3942)



L=0.004
omega=2*3.1415926*100000
j = complex(0,1)
k=omega/c

def p(x) :
    return ( cmath.exp(j*k*x) - cmath.exp(-j*k*x) ) / ( cmath.exp(j*k*L) - cmath.exp(-j*k*L) )

def u(x) :
    return -1/(rho*c) * ( cmath.exp(j*k*x) + cmath.exp(-j*k*x) ) / ( cmath.exp(j*k*L) - cmath.exp(-j*k*L) )


# The evaluation point chosen in the input file:
x = L
print "Pressure at ", x, ": ", p(x)
print "Velocity at ", x, ": ", u(x)

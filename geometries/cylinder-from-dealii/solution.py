# A little script that helps with evaluating the pressure and velocity at
# arbitrary points along the x-axis.

import cmath
import numpy as np
import matplotlib.pyplot as plt

# For the case with no attenuation
#c=343.287
#rho=1.205728

# For the case with attenuation
c=complex(291.437,47.84)
rho=complex(1.32164,-0.163955)
B=complex(134606.,6726.26)
c=cmath.sqrt(B/rho)


L=0.008
f=10000
omega=2*3.1415926*f
j = complex(0,1)


def p(x) :
    return (cmath.exp(j*omega/c*x) - cmath.exp(j*omega/c*(2*L-x)) ) / ( 1-cmath.exp(2*j*omega/c*L) )

def u(x) :
    return -1/(rho*c) * (cmath.exp(j*omega/c*x) + cmath.exp(j*omega/c*(2*L-x)) ) / ( 1-cmath.exp(2*j*omega/c*L) )




# The evaluation point chosen in the input file:
x = 0.003
print ("Pressure at ", x, ": ", p(x))
print ("Velocity at ", x, ": ", u(x))


# Finally also plot the solution:
x = np.arange(0, L, L/1000)
plt.figure()
plt.ylabel('p(x)')
y = np.array(x,dtype=complex)
for i in range(0,x.size) :
  y[i] = complex(p(x[i]))
plt.plot(x, np.real(y), '-', label="Real part of p(x)")
plt.plot(x, np.imag(y), '-', label="Imaginary part of p(x)")
plt.legend()
plt.show()

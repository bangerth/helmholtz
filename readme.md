# Overview

The executable program of this project solves the equations that describe
the acoustic response of a cavity to an applied pressure at one of a
number of ports. It then outputs the average pressure and average
(volumetric) velocity at all ports.

For a given frequency omega, the equations to be solved are the
Helmholtz equations and read
```
  -omega^2 p(x,y,z) - c^2 p(x,y,z) = 0
```
where `c` is the (possibly complex-valued) wave speed and `p` is the
pressure field we seek. The boundary conditions applied to this
equation are
```
  p = 1
```
at a "source port" and
```
  p = 0
```
at all other ports. For the remainder of the boundary, the boundary
conditions applied are
```
  n . nabla p = 0
```

The program solves these equations for a range of frequencies `omega` to compute
a frequency-dependent response of the cavity.

The underlying method used for the solution of the equation is the finite element
method (FEM).


# Input

## Calling the program

The executable (let us assume that it is called `helmholtz.exe`) is
called on the command line in the following way:
```
  helmholtz.exe <instancefolder> <outputfileprefix>
```
where `<instancefolder>` is the name of a directory into which all
output files will be written, and `<outputfileprefix>` is a string
that will be prefixed to output file names so that multiple,
concurrent runs of the same program can avoid overwriting each others'
output files.

If `<outputfileprefix>` is omitted, then the empty string will be used
as a prefix. If `<instancefolder>` is omitted, then the current
directory `"."` will be used for output files.


## The `.prm` file

All remaining parameters will be read from a file called
`helmholtz.prm` located in the `<instancefolder>` passed on the
command line as described above.

The program reads the parameter values that determine what is to be
computed from this input file.
The syntax of the `helmholtz.prm` file is as follows (the values shown
here on the right hand side of the assignment of course need to be
replaced by values appropriate to a simulation):
```
set Mesh file name                       = ./mesh.msh
set Geometry conversion factor to meters = 0.001

set Material properties file name        = air.txt

set Frequencies                          = list(5000,10000,15000)
set Evaluation points                    = -25,15,1 ; -25,15,2

set Number of mesh refinement steps      = 0
set Finite element polynomial degree     = 2
set Number of threads                    = 1
```
The first of these parameters, obviously, corresponds to the name of a
file that contains the mesh that describes the domain on which to solve the PDE (i.e., the shape
of the cavity). The file may describe either a tetrahedral or a hexahedral mesh -- both are
supported. The ending of the name of the file determines
which format it is to be read in:

  - `.msh` indicates that the file is to be interpreted as a file generated by the gmsh
    program.
  - `.mphtxt` indicates that the file is to be interpreted as a file generated by
    COMSOL multiphysics in its text file format.
  - `.inpt` indicates that the file is to be interpreted as a file generated by
    ABAQUS in its usual mesh file format (but *not* as a "FlatDeck" file).

The second parameter in this block describes a scaling
factor: mesh files generally are drawn up in community-specific units
(millimeters, inches) whereas this program internally works with SI
units (i.e., kilograms, meters, seconds) and the parameter describes
the scaling from length units used in the mesh file to meters. In the
example above, this corresponds to the mesh file using millimeters.

The second block (third parameter) describes where to find the frequency-dependent
mechanical properties of the medium. All parameters are given in SI
units. The detailed format of this file is discussed below.

The third block describes the frequencies that should be
computed. There are three possibilities for this parameter:

  - Specify `linear_spacing(f1,f2,nsteps)`: This computes `nsteps`
    frequencies between `f1` and `f2` (given in Hz) with equal spacing
    between frequencies.
  - Specify `exp_spacing(f1,f2,nsteps)`: This computes `nsteps`
    frequencies between `f1` and `f2` (given in Hz) with equal spacing
    between the logarithms of frequencies. In other words, the gaps
    between successive frequencies grow exponentially: On a log plot,
    the frequencies so computes are equall spaced.
  - Specify `list(f1,f2,f3,...)`: Here, the program computes all of
    the frequencies explicitly listed between parentheses. The list
    may have any length, as long as it contains at least one element.

The second parameter in this block also describes a list of three-dimensional points
(using the same units as the mesh file) at which the pressure and
volumetric velocity will be evaluated for each frequency. Points are
separated by semicolons, and the components of the points are
separated by commas.

The fourth block of parameters shown above describes properties of the
discretization, i.e., of _how_ the problem (formulated as a PDE)
should be solved rather than what the problem actually is. In
particular, the parameters list the number of mesh refinement steps
(each refinement step replaces each cell of the mesh by its four
children) as well as the polynomial degree of the finite element
approximation.

The number of mesh refinement steps is interpreted in the following
way: If it is positive or zero, then the mesh on which the solution is
computed is simply the mesh taken from the file listed in the first
parameter, this many times refined by replacing each cell by its four
children in each refinement step. In particular, this leads to using
the same mesh for all frequencies.

On the other hand, if the number of mesh refinement steps is negative,
then the mesh is adapted using the following algorithm. For a given
wave speed, we can compute the wavelength of oscillations for a
given frequency _w_ via
```
  lambda = c/f
         = c/(w/(2pi))
```
If `lambda` is larger than the diameter of the domain, then we replace
it by the diameter of the domain since that is the largest wavelength
at which the solution can vary.
We can then determine a frequency-adapted mesh size in the following
way: We want that there are at least _N_ grid points per wave
length. Since there are as many grid points per cell as the polynomial
degree _p_, this equations to requiring that the mesh size _Delta x_
satisfies _Delta x / p <= lambda/N_
or equivalently: _Delta x <= lambda/N * p_. The `Mesh refinement steps`
parameter is then interpreted as _N_ if it is negative.

The last parameter, `Number of threads`, indicates how many threads
the program may use at any given time. Threads are used to compute
the frequency response for different frequencies at the same time
since these are independent computations. A value of zero means that
the program may use as many threads as it pleases, whereas a positive
number limits how many threads (and consequently CPU cores) the
program will use.


## Describing the material properties of the medium

The `.prm` file mentioned above makes reference to a file that
contains the frequency-dependent material properties of the
medium. (In the example above, it is called `air.txt`.) This file
needs to have the following format:
```
%frequency(Hz)    density(kg/m3), real/imag          bulk modulus(Pa), real/imag
10                1.7507     -251.40                 112615.491      119.669
60.201            1.7507     -41.763                 112631.842      720.021
110.402           1.7507     -22.776                 112671.498      1318.65
160.603           1.7506     -15.66                  112734.259      1914.15
210.804           1.7505     -11.934                 112819.809      2505.12
261.005           1.7504     -9.6422                 112927.724      3090.22
311.206           1.7502     -8.0903                 113057.471      3668.13
361.407           1.75       -6.97                   113208.422      4237.61
[...]
```
The first line of this file is treated as a comment and
ignored. Following lines provide the frequency at the beginning of each
row of the table, followed by real and imaginary parts of the density and bulk
modulus.

The file as a whole needs to end in a newline.

When asked to compute the response of a cavity at a specific
frequency, the solver linearly interpolates between lines of the
file. For example, at a frequency of 185.7035 Hz, it will use density and
bulk modulus values halfway between the ones tabulated for 160.603 and
210.804 Hz. If the solver is asked for a frequency below the very
first frequency provided in the file, it simply uses the values of the
first value provided. Similarly for a frequency above the last one
provided: It just takes the last provided values. This allows to
specify air in the following way, using only a single line, given that
air has no substantial frequency dependence to its material properties:
```
%         frequency(Hz)        density(kg/m3), real/imag      bulk modulus(Pa), real/imag
          10                   1.205728     0                 142090.344491053     0       
```


# Output

The output of the program consists of three pieces:
- the frequency response file in human-readable form
- the frequency response file in machine-readable form
- and a number of files in the visualization directory.

### The file `<outputfileprefix>frequency_response.txt`

After each frequency has been computed, the program appends results to
a frequency response file.

The principal piece of interest in this file is the `M` matrix. For a geometry
with `n=3` ports, this matrix is defined by stating that
```
|  m11   m12   m13   m14   m15   m16|   | P1 | = |b1|
|  m21   m22   m23   m24   m25   m26|   | U1 |   |b2|
|  m31   m32   m33   m34   m35   m36|   | P2 |   |b3|
                                        | U2 |
                                        | P3 |
                                        | U3 |
```
where the components `b1`, `b2, `b3` of the right hand side vector correspond
to volumetric source terms and are zero for our current situation. For a general
case of `n` ports, the `M` matrix has size `n` times `2*n`.

In the equation above, `Pi` and `Ui` are the average pressure and normal
velocity at port `i`, where velocities are computed *into* the geometry
and the average is computed by forming the integral of the pressure or
velocity over the area of a port divided by the area of the port.

The way the program computes the `M` matrix is by solving for situations
where one prescribes a unit presure on one port and zero pressure on
all other ports, and then computing the average velocity on all ports. For
example, if one prescribes a unit pressure on port 1, then we know that
```
|  m11   m12   m13   m14   m15   m16|   |  1  | = |0|
|  m21   m22   m23   m24   m25   m26|   | U11 |   |0|
|  m31   m32   m33   m34   m35   m36|   |  0  |   |0|
                                        | U21 |
                                        |  0  |
                                        | U31 |
```
and `U11`, `U12`, and `U13` are known. We can repeat this for all three
ports, and this leads to the linear system
```
|  m11   m12   m13   m14   m15   m16|   |  1   0   0  | = |0 0 0|
|  m21   m22   m23   m24   m25   m26|   | U11 U12 U13 |   |0 0 0|
|  m31   m32   m33   m34   m35   m36|   |  0   1   0  |   |0 0 0|
                                        | U21 U22 U23 |
                                        |  0   0   1  |
                                        | U31 U32 U33 |
```
Here, this means that we have 9 equations for the 18 entries of the matrix `M`.
In general, we will have `n*n` equations for the `n*(2n)` entries of `M`. That
means, we can choose `n*n` entries of `M` at will, and compute the rest from the
equations we have. We will do this by choosing the elements of every other column
of `M` in a convenient way, namely
```
|  m11   -1   m13    0   m15    0|   |  1   0   0  | = |0 0 0|
|  m21    0   m23   -1   m25    0|   | U11 U12 U13 |   |0 0 0|
|  m31    0   m33    0   m35   -1|   |  0   1   0  |   |0 0 0|
                                     | U21 U22 U23 |
                                     |  0   0   1  |
                                     | U31 U32 U33 |
```
as this will then allow us to put all of the terms involving `Uij`
onto the right hand side and we obtain
```
|  m11   m13   m15|   |  1   0   0 | = |U11 U12 U13|
|  m21   m23   m25|   |  0   1   0 |   |U21 U22 U23|
|  m31   m33   m35|   |  0   0   1 |   |U31 U32 U33|
```
This then immediately allows us to read off the entries of the
matrix on the left, and we get that the matrix we seek is of the
form
```
    |  U11   -1   U13    0   U15    0|
M = |  U21    0   U23   -1   U25    0|
    |  U31    0   U33    0   U35   -1|
```
with the obvious generalization to arbitrary numbers of ports.


*TODO:* point values


### The file `<outputfileprefix><freq#>frequency_response.csv`

For each frequency computed, the program also generates a separate file with
results in computer-readable, CSV format.

*TODO:* Update with the exact contents of these files once settled upon.


### The file `output.log`

This file contains some status information that chronicles progress of what
the program is doing. It will look similar to the following:
```
INFO Program started with argument '.'
INFO Material parameters file contains data for 200 frequencies ranging from 10 to 10000Hz.
INFO Number of frequencies scheduled: 1
INFO Reading mesh file <../helmholtz/geometries/cylinder-from-dealii/tet.msh> in GMSH .msh format
INFO The mesh has 30720 cells
INFO Found boundary ids 0 1 2 
INFO The mesh has 5561 unknowns
INFO Computing data for omega=628319, source port boundary id=1
INFO Computing data for omega=628319, source port boundary id=2


+---------------------------------------------+------------+------------+
| Total wallclock time elapsed since start    |      2.37s |            |
|                                             |            |            |
| Section                         | no. calls |  wall time | % of total |
+---------------------------------+-----------+------------+------------+
| Assemble linear system          |         2 |     0.092s |       3.9% |
| Creating visual output          |         2 |     0.582s |        25% |
| Make grid                       |         1 |     0.148s |       6.3% |
| Postprocess                     |         2 |    0.0118s |       0.5% |
| Set up system                   |         1 |    0.0187s |      0.79% |
| Solve linear system             |         2 |      1.52s |        64% |
+---------------------------------+-----------+------------+------------+
```
The information in this file is mostly interesting to see what is currently
happening and to obtain some statistics on the mesh being used.


### Monitoring progress

The `frequency_response.txt` and `frequency_response.csv` files are
updated every time the program has finished computing the response of
the cavity for a particular frequency. As a consequence, the file
contains a record of all computed frequencies.

To monitor the progress of computations -- for example for displaying
a progress bar -- open the `frequency_response.csv` file periodically
(say, once a second) and read what's in it. If all you want is to show
progress: The first line of the file has a format that, during
computations, says something like "`# 42/100 frequencies computed`",
allowing for a quick overview where computations currently are. If you
want something fancier, you can actually parse the contents of the
file and update a graph of the frequency-dependent cavity response
every time you read through the file. This way, the graph will fill in
over time.

*TODO:* Update with the exact contents of these files once known.

When the program has successfully finished all computations, it
generates a file `<outputfileprefix>success_signal.txt` in the
instance folder, and terminates. If the solver fails for some reason,
it instead generates a file
`<outputfileprefix>solver_failure_signal.txt` in the instance folder.
Both of these files are removed at the beginning of the program run
should they exist.


### The directory `<outputfileprefix>visualization/`

This directory contains one file for each input frequency and source port, with each file providing
all of the information necessary to visualize the solution. The format used for these
files is VTU, and the solution can be visualized with either the
[Visit](https://wci.llnl.gov/simulation/computer-codes/visit) or
[Paraview](https://www.paraview.org/) programs.
The file names are of the form `<outputfileprefix>visualization/solution-XXXXX.YY.vtu` where `XXXXX`
denotes the (integer part of the) frequency (in Hz) at which the solution
is computed, and `YY` is the number of the source port.

For debugging purposes, it is often useful to also see the geometry of
the domain and how different parts of the boundary are labeled by the
boundary indicators that are used to identify "ports". To this end,
the program writes a file
`<outputfileprefix>visualization/surface.vtu` right after reading in
the mesh and before anything else happens -- that is, before much of
anything else can go wrong in the program.

When visualized, the data stored in this file might look like the
following picture for one of the geometries with which this program
was tested:

![xxx](geometries/abaqus-geometry/boundary_ids.png)

The image shows how most of the boundary of the geometry is labeled
with boundary indicator zero (which the program interprets as "not a
port"), whereas the left side and a thin strip along the top right
edge are labeled with boundary indicators two and one, respectively
(which the program then interprets as the "ports" of the geometry).


# Error reporting

There are many ways execution of this program can run into errors. The most common are
that something is wrong with the input: The file that is listed as containing the mesh
does not exist, or it may not follow the structure required of this file; the file that
describes the speed of sound does not exist, or it has invalid values (say, negative
wave speeds). But there are also ones that could indicate a bug in the code, or that
the program has run out of resources (say, it can't allocate enough memory).

All of these errors lead to two consequences:

* The program terminates with a non-zero return code.
* The program leaves behind a file called `<instance-folder>/<output-prefix>/solver_failure_signal.txt`.
* The program writes the error message into a file 
  `<instance-folder>/<output-prefix>/error.log`. These error messages look similar to the following one
  that resulted from an mesh input file that did not conform to the specifications:
```
    ERROR Exception while computing for frequency 100000:

    --------------------------------------------------------
    An error occurred in line <2109> of file
    </home/bangerth/p/deal.II/1/dealii/source/grid/grid_in.cc> in function
         void dealii::GridIn<dim, spacedim>::read_msh(std::istream&) [with int dim = 3; int spacedim = 3; std::istream = std::basic_istream<char>]
    The violated condition was:
         cells.size() > 0
    Additional information:
         While reading a gmsh file, the reader function did not find any cells.
         This sometimes happens if the file only contains a surface mesh, but
         not a volume mesh.

         The reader function did find 0 lines and 167936 facets (surface
         triangles or quadrilaterals).
    --------------------------------------------------------
```

# Terminating execution

There may be times where callers of this program do not want it to continue with
its computations. In this case, an external program should place the text `STOP`
into a file called `<outputfileprefix>termination_signal.txt` in the
instance folder. This will terminate the program. That said, because the program outputs all
data already computed whenever one frequency is finished, even when a
program execution is terminated, already computed information is still
stored in the output files.

The program works on input frequencies in an unpredictable order, since work
for each frequency is scheduled with the operating system at the beginning
of program execution, but it is the operating system's job to allocate CPU
time for each of these tasks. This is often done in an unpredictable order.
As a consequence, the frequencies already worked on at the time of termination
may or may not be those listed first in the input file.


# How this program was tested

We can check for the correctness of the program using the following
set up:

The mesh is a cylinder along the _x_-axis with length 4 and radius
1. We use a scaling factor of 0.001, so this corresponds to a length
_L_=4mm and radius _r_=1mm.

The solution to this problem is one-dimensional, and reads
```
  p(x) = a*exp(j*k*x) + b*exp(-j*k*x)
```
with _k=omega/c_ and _a,b_ so that
```
  p(0) = 1
  p(L) = 0
```
which implies
```
  a+b=1
  a*exp(j*omega/c*L) + b*exp(-j*omega/c*L)=0
```
Inserting the first of these equations into the second, and multiplying by
`exp(j*omega/c*L)`, we get
```
  (1-b)*exp(2*j*omega/c*L) + b=0
```
which implies
```
  exp(2*j*omega/c*L) + (1-exp(2*j*omega/c*L))*b=0
```
and consequently
```
  b = -exp(2*j*omega/c*L) / [ 1-exp(2*j*omega/c*L) ]
  a = (1-b)
    = 1+exp(2*j*omega/c*L) / [ 1-exp(2*j*omega/c*L) ]
    = 1 / [ 1-exp(2*j*omega/c*L) ]
```
Taken together, this results in the following solution:
```
  p(x) = exp(j*omega/c*x)/[ 1-exp(2*j*omega/c*L) ]
         - exp(2*j*omega/c*L) * exp(-j*omega/c*x)/[ 1-exp(2*j*omega/c*L) ]
       = [ exp(j*omega/c*x) - exp(j*omega/c*(2L-x)) ]
           / [ 1-exp(2*j*omega/c*L) ]
```
We can easily verify that `p(0)=1` and `p(L)=0`, as expected.

Furthermore, we can compute the (volumetric) velocity as
```
  u(x) = -1/(j rho omega) nabla p
```
of which only the _x_-component is nonzero and reads as
```
  u_x(x) = -1/(rho c) [ exp(j*omega/c*x) + exp(j*omega/c*(2L-x)) ]
           / [ 1-exp(2*j*omega/c*L) ]
```
So, at the left end of the cylinder, we have
```
  u_x(0) = -1/(rho c) [ 1 + exp(2*j*omega/c*L)) ]
           / [ 1-exp(2*j*omega/c*L) ]
```
and at the right end
```
  u_x(L) = -2/(rho c) exp(j*omega/c*L) / [ 1-exp(2*j*omega/c*L) ]
```

We can test all of this for a concrete choice of wave speed `c`,
density `rho`, and frequency `omega = 2*pi*f` using the cylinder
of length `L=4mm` described above. Using the following input file,
```
set Mesh file name                       = ../helmholtz/geometries/cylinder/cylinder-2-times-refined.msh
set Geometry conversion factor to meters = 0.001
set Evaluation points                    = 1,0.5,0.5

set Material properties file name       = ../helmholtz/material-models/air.txt

set Frequencies                          = list(100000)
set Number of mesh refinement steps      = 0
set Finite element polynomial degree     = 1
set Number of threads                    = 1
```
we have `rho=1.18 kg/m^3`, `c=343 m/s`, and `f=100 kHz` (which
produces a wave length of 3.43mm, just short of the length of the
cylinder). A visualization of the real part of the pressure solution
looks as follows, on a hexahedral mesh with 11,121 unknowns:

![xxx](geometries/cylinder/100kHz-hex.png)

For this set of frequencies, we then expect
```
  u_x(0) = 0-0.00144j    # 
```
and at the right end
```
  u_x(L) = 0-0.00286j    # 
```
where at the right end, the sign needs to be flipped to obtain the
_inward_ velocity. Indeed, on the mesh shown above, the program
produces these values for the two ports:
```
  -0.00146963j
  +0.00290720j
```
This matches to good accuracy the expected values, and a further
refined mesh will result in even better accuracy.


The results above are obtained with a purely real wave speed (loss
angle zero, no attenuation).
```
set Mesh file name                       = ../helmholtz/geometries/cylinder/cylinder-2-times-refined.msh
set Geometry conversion factor to meters = 0.001
set Evaluation points                    = 1,0.5,0.5

set Material properties file name       = ../helmholtz/material-models/air.txt

set Frequencies                          = list(100000)
set Number of mesh refinement steps      = 0
set Finite element polynomial degree     = 1
set Number of threads                    = 1
```
With these values, we then get the following expected values:
```
  u_x(0) = 0.002238 - 0.000752j
```
and at the right end
```
  u_x(L) = 0.0005134 - 0.00125j
```
On the same mesh as shown in the previous figure, the obtained values
for the inward volumetric velocities are
```
   0.00226466 -0.000784016j
  -0.000514756+0.001271710j
```
which is again a decent approximation after accounting for having to
switch the sign on the second value.


The experiments above were obtained on a hexahedral mesh, which
deal.II can produce internally but which are typically difficult to
generate with gmsh at high quality. Rather, gmsh generates meshes such
as this one:

![xxx](geometries/cylinder/100kHz-2-times-refined.png)

On this mesh (with 8,373 unknowns), we obtain the following data for the case without attenuation:
```
  0-0.00157234j
  0+0.00275222j
```
And with attenuation:
```
  0.002103030-0.00100064j
 -0.000452395+0.00121371j
```
This is not quite as accurate, though not a terrible approximation
either. That said, we can ask gmsh to refine the mesh
once more. The solution then looks like this, on a mesh with 60,777
unknowns:

![xxx](geometries/cylinder/100kHz-3-times-refined.png)

Now the corresponding values are
```
  0-j*0.00155675
  0+j*0.00282744
```
without attenuation, and
```
  0.002172990-0.000892128j
 -0.000500042+0.001238410j
```
with attenuation. These values are now substantially closer to the analytically
obtained (correct) values shown above.


Finally, the input files above allow the evaluation of point values
for pressure and velocity. At the point of our choice, `(1, 0.5,
0.5)`, we can evaluate the pressure as
```
  p(1)   = -0.8297662828 - 0.4150015992e-6j
  u_x(1) = -0.266e-8     - 0.001775594289j
```
For the coarser of the tetrahedral meshes,
we obtain
```
  pressure = -0.763803+0j
  u_x      = -0.00212938j
```
whereas for the finer mesh, we get
```
  pressure = -0.810079+0j
  u_x      = -0.00178777j
```
As before the values on the finer mesh are reasonably close to the
analytical values.

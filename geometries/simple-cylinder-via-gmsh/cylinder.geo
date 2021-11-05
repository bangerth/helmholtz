SetFactory ("OpenCASCADE"); // use OpenCASCADE kernel

General.MessageFontSize=20;  // i can't read the messages in the gmsh window, so increase the font


// size of the cylinder
R=2;  // radius
L=5;  // length

Icyl=newv;   // create a new volume index
Cylinder(Icyl) = {0,0,0,  0,0,L, R};  // create the cylinder

// inspecting the cylinder in the gmsh gui, the surface on one end of the
// cylinder was automatically given the index 2, and the other was given 3,
// so we'll create two named surfaces called "port1" and "port2" to denote the
// two ends of the cylinder
Physical Surface("port1") = {2};  
Physical Surface("port2") = {3};

// perform the 1,2 and 3 dimensional meshing
Mesh 1;
Mesh 2;
Mesh 3;

// by default, onlyl the mesh associated with physical groups get saved, so the 
// surface of the cylinder and the interior volume of the cylinder arn't saved to 
// the .msh file.  by setting "SaveAll" to 1, it seems to be saving these additional
// elements.  however i haven't verified this by actually reading in the mesh
Mesh.SaveAll=1; 


// save the mesh
Save "script6_simple_cylinder_for_wolfgang.msh";


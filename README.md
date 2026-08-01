# Introduction and motivation
This project contains a python code into FEniCSx environment to solve a potential flow with a Finite Element discretization of the Laplace equation.


The project has been created as a validation of personal studies on how to properly write a weak form of a differential problem and solve it. 

The Laplace equation for the velocity potential of an incompressible, irrotational and inviscid flow offers a simple example to test.

# Requirements
Conda must be installed on the machine where the code is run

# How to run it
1. Move to the project folder

2. Run the following command in the terminal to create and setup the local environment in the project folder:
conda env create -f environment.txt --prefix ./venv

3. Run the following command in the terminal to activate the local environment:
conda activate ./venv

4. Run the code src/fenicsx_potential_flow_2d/main.py

# Inputs
in src/fenicsx_potential_flow_2d/main.py:
- obstacles (circles) center locations and radiuses
- domain (rectangular) size
- mesh size
- input velocity

# Output
- plot relative pressure field contour
- plot streamlines
- plot velocity vector fiels

# Boundary conditions
- Dirichlet on left boundary, imposing infinite velocity directed towards positive x (from left to right)
- Neumann on right, imposing normal component of velocity equal to infinite velocity
- Natural Neumann (null normal component of velocity, i.e., no penetration) on top, bottom and obstacles, naturally taken into account by excluding these boundaries in the right hand side of the weak formulation (L)

# Mesh 
- quadratic quadrilateral


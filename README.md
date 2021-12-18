# eecs106_dresspack_sim
It is for eecs106 final project
TubeDyn
Learning to systhesis tube dynamics

Install
Install anaconda and create a new virtual environment
Download the Isaac Gym source code from here and extract to ./isaacgym
Install torch 1.8.0 with suitable cuda version as here
cd isaacgym/python/
pip install -e .
Run cd examples and python joint_monkey.py to test the installation
pip install -r requirements.txt
cd simulations and test FEM simulation with python sim_biotac.py
Generate Soft Model (Optional)
Install FreeCAD
Create a tube and save as .stl
Install fTetWild
Convert .stl to .mesh
Convert .mesh to .tet with simulations/mesh2tet.py
References
Robot dress pack simulation paper_1
TODO
Modify the robot configuration (i.e., add two robots and correctly add the tube)
Modify the robot dynamical parameters (i.e., damping, stiffness, limits, etc) to make it realistic
Modify the robot control method (refer to all franka related .py)

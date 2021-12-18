# TubeDyn
Learning to systhesis tube dynamics

## Install
- Install [anaconda](https://docs.anaconda.com/anaconda/install/linux/#installation) and [create a new virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
- Download the Isaac Gym source code from [here](https://developer.nvidia.com/isaac-gym/download) and extract to `./isaacgym`
- Install torch 1.8.0 with suitable cuda version as [here](https://pytorch.org/get-started/previous-versions/#linux-and-windows-1)
- `cd isaacgym/python/`
- `pip install -e .`
- Run `cd examples` and `python joint_monkey.py` to test the installation
- `pip install -r requirements.txt`
- `cd simulations` and test FEM simulation with `python sim_biotac.py` 

## Generate Soft Model (Optional)
- Install [FreeCAD](https://www.freecadweb.org/downloads.php)
    - Create a [tube](https://wiki.freecadweb.org/Part_Tube) and save as .stl
- Install [fTetWild](https://github.com/wildmeshing/fTetWild#installation-via-cmake) 
    - Convert .stl to .mesh
- Convert .mesh to .tet with `simulations/mesh2tet.py`

## References
- Robot dress pack simulation [paper_1](http://www.diva-portal.org/smash/get/diva2:1222002/FULLTEXT01.pdf)

## TODO
- Modify the robot configuration (i.e., add two robots and correctly add the tube)
- Modify the robot dynamical parameters (i.e., damping, stiffness, limits, etc) to make it realistic
- Modify the robot control method (refer to all franka related `.py`)

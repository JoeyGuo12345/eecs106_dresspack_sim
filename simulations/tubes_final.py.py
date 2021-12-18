import os
import math
import time
import copy
import random
import argparse
import numpy as np
from shutil import copyfile
import fileinput
from scipy.spatial.transform import Rotation as R
from kinematics import lrmate_fk
from kinematics import lrmate_ik
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from collections import deque
import itertools

import torch
class controller(object):
    def __init__(self, Kp, Ki, Kd, Kw, limb):
        """
        Constructor:

        Inputs:
        Kp: 7x' ndarray of proportional constants
        Ki: 7x' ndarray of integral constants
        Kd: 7x' ndarray of derivative constants
        Kw: 7x' ndarray of antiwindup constants
        limb: baxter_interface.Limb or sawyer_interface.Limb
        """


        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._Kw = Kw

        self._LastError = np.zeros(len(Kd))
        self._LastTime = 0;
        self._IntError = np.zeros(len(Ki))
        self._ring_buff_capacity = 3
        self._ring_buff = deque([], self._ring_buff_capacity)

        self._curIndex = 0;
        self._maxIndex = 0;

        self._limb = limb

        # For Plotting:
        self._times = list()
        self._actual_positions = list()
        self._actual_velocities = list()
        self._target_positions = list()
        self._target_velocities = list()

def create_sim(gym, gpu_physics=0, gpu_render=0, frict_coeff=7.83414394e-01):
    """Set the simulation parameters and create a Sim object."""
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0e-4  # Control frequency
    sim_params.substeps = 1  # Physics simulation frequency (multiplier)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    sim_params.stress_visualization = True  # von Mises stress
    sim_params.stress_visualization_min = 0.0
    sim_params.stress_visualization_max = 1.0e9

    sim_params.flex.solver_type = 5  # PCR (GPU, global)
    sim_params.flex.num_outer_iterations = 8
    sim_params.flex.num_inner_iterations = 40
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
    sim_params.flex.deterministic_mode = True

    sim_params.flex.geometric_stiffness = 1.0
    sim_params.flex.shape_collision_distance = 2e-3  # Distance to be maintained between soft bodies and other bodies or ground plane
    sim_params.flex.shape_collision_margin = 1e-3  # Distance from rigid bodies at which to begin generating contact constraints

    sim_params.flex.friction_mode = 2  # Friction about all 3 axes (including torsional)
    sim_params.flex.dynamic_friction = frict_coeff

    sim = gym.create_sim(gpu_physics, gpu_render, sim_type, sim_params)
    
    return sim

def set_asset_options():
    """Set asset options common to all assets."""

    options = gymapi.AssetOptions()
    options.flip_visual_attachments = False
    options.armature = 0.0
    options.thickness = 0.0 # 0.01
    options.linear_damping = 0.0
    options.angular_damping = 0.0
    options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    options.min_particle_mass = 1e-20

    return options

def set_tube_matl_props(base_dir, type, resolution, density, elast_mod, poiss_ratio):
    """Set the BioTac material properties by copying and modifying a URDF template."""
    template_path = os.path.join(base_dir, f'{type}_template.urdf')
    file_path = os.path.join(base_dir, f'{type}.urdf')
    copyfile(template_path, file_path)
    with fileinput.FileInput(file_path, inplace=True) as file_obj:
        for line in file_obj:
            if 'density' in line:
                print(line.replace('density value=""', f'density value="{density}"'), end='')
            elif 'sparse' in line:
                print(line.replace('sparse/tube.tet', f'{resolution}/tube.tet'), end='')
            elif 'youngs' in line:
                print(line.replace('youngs value=""', f'youngs value="{elast_mod}"'), end='')
            elif 'poissons' in line:
                print(line.replace('poissons value=""', f'poissons value="{poiss_ratio}"'), end='')
            else:
                print(line, end='')

def load_assets(gym, sim, base_dir, object, options, fix=True, gravity=False):
    """Load assets from specified URDF files."""
    options.fix_base_link = True if fix else False
    options.disable_gravity = True if not gravity else False
    handle = gym.load_asset(sim, base_dir, object + '.urdf', options)
    
    return handle

def set_scene_props(num_envs=1, env_dim=1.5):
    """Set the scene and environment properties."""

    envs_per_row = int(np.ceil(np.sqrt(num_envs)))
    env_lower = gymapi.Vec3(-env_dim, 0, -env_dim)
    env_upper = gymapi.Vec3(env_dim, env_dim, env_dim)
    scene_props = {'num_envs': num_envs,
                   'per_row': envs_per_row,
                   'lower': env_lower,
                   'upper': env_upper}

    return scene_props

def create_scene(gym, sim, props, assets_tube):
    """Create a scene (i.e., ground plane, environments, connector actors)."""

    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    env_handles = []
    actor_handles_tube = []
    up_rigid_body_handles = []
    down_rigid_body_handles = []
    
    for i in range(props['num_envs']):
        env_handle = gym.create_env(sim, props['lower'], props['upper'], props['per_row'])
        env_handles.append(env_handle)

        pose = gymapi.Transform()
        collision_group = i
        collision_filter = 0

        pose.p = gymapi.Vec3(0.0, 1.5, 0.0)
        r = R.from_euler('XYZ', [0, 0, 0], degrees=True)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        
        actor_handle_tube = gym.create_actor(env_handle, assets_tube, pose, f"tube_{i}", collision_group, collision_filter)
        actor_handles_tube.append(actor_handle_tube)

        up_body_handler = gym.get_rigid_handle(env_handle, f"tube_{i}", 'up')
        down_body_handler = gym.get_rigid_handle(env_handle, f"tube_{i}", 'down')
        up_rigid_body_handles.append(up_body_handler)
        down_rigid_body_handles.append(down_body_handler)

    return env_handles, actor_handles_tube, up_rigid_body_handles, down_rigid_body_handles

def create_viewer(gym, sim):
    """Create viewer and axes objects."""

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 5.0
    camera_props.width = 1920
    camera_props.height = 1080
    viewer = gym.create_viewer(sim, camera_props)
    camera_pos = gymapi.Vec3(-13, 13, -15)
    camera_target = gymapi.Vec3(0.75, 1.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

    axes_geom = gymutil.AxesGeometry(0.1)

    return viewer, axes_geom

def sample_traj(q_vec):
    """
    TODO
    Sample trajectories for two connectors, including x_ref, v_ref
    """
    T_i_w = np.eye(4)
    T_i_w = np.array([[0, -1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    # T_i_w[0:3, 0:3] = R.from_euler('xyz', [0, 90, 90], degrees=True).as_matrix()
    T_w_i = np.linalg.inv(T_i_w)

    # find position and angle of each joint
    fk = lrmate_fk(q_vec)

    # find the two joints where dresspack is attached to
    up = fk[0]
    down = fk[1]
    #up = np.concatenate((up[0:3], np.asarray(get_quaternion_from_euler(up[3], up[4], up[5]))))
    #down = np.concatenate((down[0:3], np.asarray(get_quaternion_from_euler(down[3], down[4], down[5]))))
    up[1] += 1.5
    down[1] += 1.5
    down[4] += 90

    up_mat = np.eye(4)
    up_mat[0:3, 3] = up[0:3]
    up_mat[0:3, 0:3] = R.from_euler('xyz', up[3:], degrees=True).as_matrix()
    up_mat = up_mat @ T_w_i
    up = np.zeros_like(up)
    up[0:3] = up_mat[0:3, 3]
    up[3:] = R.from_matrix(up_mat[0:3, 0:3]).as_euler('xyz', degrees=True)

    down_mat = np.eye(4)
    down_mat[0:3, 3] = down[0:3]
    down_mat[0:3, 0:3] = R.from_euler('xyz', down[3:], degrees=True).as_matrix()
    down_mat = down_mat @ T_w_i
    down = np.zeros_like(down)
    down[0:3] = down_mat[0:3, 3]
    down[3:] = R.from_matrix(down_mat[0:3, 0:3]).as_euler('xyz', degrees=True)

    return np.asarray([up, down])


def extract_nodal_coords(gym, sim, particle_states):
    """Extract the nodal coordinates for the BioTac from each environment."""

    gym.refresh_particle_state_tensor(sim)
    num_envs = gym.get_env_count(sim)
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles / num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env
        local_particle_index = global_particle_index % num_particles_per_env
        nodal_coords[env_index][local_particle_index] = pos.numpy()

    return nodal_coords

def hardcode_sample_traj(sim, gym, T_up, T_down):
    """
    TODO
    Sample trajectories for two connectors, including x_ref, v_ref
    """
    init_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).cpu().numpy()
    init_up, init_down = init_state[0], init_state[1]
    init_up_mat, init_down_mat = np.eye(4), np.eye(4)
    init_up_mat[0:3, 3], init_down_mat[0:3, 3] = init_up[0:3], init_down[0:3]
    init_up_mat[0:3, 0:3] = R.from_quat(init_up[3:7]).as_matrix()
    init_down_mat[0:3, 0:3] = R.from_quat(init_down[3:7]).as_matrix()
    init_up_mat = init_up_mat @ T_up
    init_down_mat = init_down_mat @ T_down
    T_up, T_down = np.zeros(6), np.zeros(6)
    T_up[0:3], T_down[0:3] = init_up_mat[0:3, 3], init_down_mat[0:3, 3]
    T_up[3:], T_down[3:] = R.from_matrix(init_up_mat[0:3, 0:3]).as_euler('xyz', degrees=True), R.from_matrix(init_down_mat[0:3, 0:3]).as_euler('xyz', degrees=True)

    return np.asarray([T_up, T_down])


def get_quaternion_from_euler(yaw, pitch, roll):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]

def control_connectors(gym, sim, envs, trajs, args, step, prev_state, dt, last_error, last_target):
    """
    TODO
    PD control for the position/rotation of the connector
    1. Get reference position x_ref and reference velocity v_ref from trajs
    2. Get current position x_curr and current velocity v_curr from the simulation
    3. Compute velocity command v = (x_ref - x_curr) * k + (v_ref - v_curr) * d
    4. Execute v
    In 3D world with linear and angular velocity
    
    OR
    Maybe just use set_rigid_body_state_tensor to set the position/quat of the objects, not sure whether this works
    """
    gym.refresh_rigid_body_state_tensor(sim)
    state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))

    """ position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]) 
    0 for up (on the left) and 1 for down (on the right)"""
    # Set position

    target = last_target + step
    tar = np.zeros((2,7))
    tar[0] = np.concatenate((target[0][0:3], np.asarray(get_quaternion_from_euler(target[0][3], target[0][4], target[0][5]))))
    tar[1] = np.concatenate(
        (target[1][0:3], np.asarray(get_quaternion_from_euler(target[1][3], target[1][4], target[1][5]))))

    curr_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).cpu().detach().numpy()[:, 0:6]
    error = last_target - curr_state
    err = np.zeros((2, 7))
    err[0] = np.concatenate(
        (error[0][0:3], np.asarray(get_quaternion_from_euler(error[0][3], error[0][4], error[0][5]))))
    err[1] = np.concatenate(
        (error[1][0:3], np.asarray(get_quaternion_from_euler(error[1][3], error[1][4], error[1][5]))))

    kp = 1
    state[0, 0] = tar[0][0]# + kp * err[0][0]
    state[0, 1] = tar[0][1]# + kp * err[0][1]
    state[0, 2] = tar[0][2]# + kp * err[0][2]
    state[0, 3] = tar[0][3]# + kp * err[0][3]
    state[0, 4] = tar[0][4]# + kp * err[0][4]
    state[0, 5] = tar[0][5]# + kp * err[0][5]
    state[0, 6] = tar[0][6]# + kp * err[0][6]


    state[1, 0] = tar[1][0]# + kp * err[1][0]
    state[1, 1] = tar[1][1]# + kp * err[1][1]
    state[1, 2] = tar[1][2]# + kp * err[1][2]
    state[1, 3] = tar[1][3]# + kp * err[1][3]
    state[1, 4] = tar[1][4]# + kp * err[1][4]
    state[1, 5] = tar[1][5]# + kp * err[1][5]
    state[1, 6] = tar[1][6]# + kp * err[1][6]


    # # Set velocity
    # state[0, 7] = 20 * np.sin(step * np.pi / 36)
    # state[1, 7] = - 20 * np.sin(step * np.pi / 36)


    # up_pos = trajs[0][0:3]
    # up_ang = trajs[0][3:6]
    # down_pos = trajs[1][0:3]
    # down_ang = trajs[1][3:6]

    state = gymtorch.unwrap_tensor(state)
    gym.set_rigid_body_state_tensor(sim, state)
    return error, target
        

def extract_results(gym, sim, envs, particle_states):
    """
    TODO 
    Extract nodal positions and forces from the simulation.
    Check sim_biotac.py
    """
    return extract_nodal_coords(gym=gym, sim=sim, particle_states=particle_states)  # (num_env x num_nodes x 3)

def set_speed(trajs, i):
    return np.sqrt((trajs[i + 1][0][0] - trajs[i][0][0]) ** 2 + (trajs[i + 1][0][1] - trajs[i][0][1]) ** 2 + (
            trajs[i + 1][0][2] - trajs[i][0][2]) ** 2)

def run_sim_loop(gym, sim, envs, viewer, actor_handles_tubes, up_rigid_body_handles, down_rigid_body_handles, axes, args):
    results = []
    
    # Get particle state tensor and convert to PyTorch tensor
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)
    biotac_state_init = copy.deepcopy(particle_state_tensor)


    # TODO sample a new trajecotory for the connectors with several waypoints

    # mount dresspack to robot
    scale = 500
    i = 0
    last_time = 0

    step = 0

    while not gym.query_viewer_has_closed(viewer):

        # Run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Visualize motion and deformation
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        gym.clear_lines(viewer)

        # TODO Control the connectors to new positions
        if step == 0 and i == 0:
            T_up, T_down = np.eye(4), np.eye(4)
            T_up[0:3, 0:3] = R.from_euler('z', -90, degrees=True).as_matrix()
            T_down[0:3, 0:3] = R.from_euler('z', 90, degrees=True).as_matrix()
            T_down[0:3, 3] -= np.array([0, -0.07, 0.77])
            prev_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).cpu().detach().numpy()[:, 0:6].copy()
            trajs = [prev_state, hardcode_sample_traj(sim, gym, T_up, T_down), hardcode_sample_traj(sim, gym, T_up, T_down)]

            offset = [[0, -0.196, 0, 0, 0, 0]]
            for o in offset:
                T_up[0:3, 3] += o[0:3]
                T_down[0:3, 3] += o[3:]
                traj = hardcode_sample_traj(sim, gym, T_up, T_down)
                trajs += [traj, traj]

            # trajs = [prev_state, , hardcode_sample_traj(sim, gym, T_up, T_down)]
            #, sample_traj([0, 0, 0, 0, -80, 0]), sample_traj([0, 0, 0, 80, -80, 0]),
            # sample_traj([0, 0, 0, 60, -80, 0]), sample_traj([0, 0, 0, 80, -80, 0])]
            #length = set_speed(trajs, i)
            last_error = 0
            last_target = trajs[0]

        step += 1
        t = gym.get_elapsed_time(sim)
        dt = t - last_time
        x = 1/scale*(trajs[i+1] - trajs[i])
        #y = trajs[i]
        last_error, last_target = control_connectors(gym, sim, envs, trajs[i+1], args, 1/scale*(trajs[i+1] - trajs[i]), prev_state, dt, last_error, last_target)
        last_time = t

        if step % 10 == 0:
            print(step, scale)



        if step >= scale:
            i += 1
            if i == len(trajs)-1:
                print("all trajectory finished")





                break
            prev_state = trajs[i]

            #length = set_speed(trajs, i)
            cur_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
            print(cur_state)
            print(step)
            step = 0
            print(trajs[i])
            print('one trajectory finished')

        
        # TODO extract results
        result = extract_results(gym, sim, envs, particle_state_tensor)
        np.save('position.npy', result)
    
    # Simulate and visualize one final step
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    gym.clear_lines(viewer)

    return results
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='tube_32x40', type=str, help='Type of the tube. Choice of tube_47x54, tube_64x80, tube_91x106, tube_126x146')
    parser.add_argument('--resolution', default='sparse', type=str, help='Resolution of the tube. Choice of sparse, density')
    
    parser.add_argument('--density', default=2e+3, type=float, help="Density of the tube")
    parser.add_argument('--elast_mod', default=1e10, type=float, help="Elastic modulus of the tube [Pa]")
    parser.add_argument('--poiss_ratio', default=0.3*3.16454280e-01, type=float, help="Poisson's ratio of the tube")
    parser.add_argument('--frict_coeff', default=2*7.83414394e-01, type=float, help='Coefficient of friction between the tube and envs')
    parser.add_argument('--err_tol', default=1.0e-4, type=float, help='Maximum acceptable error [m] between target and actual position of connector')
    
    parser.add_argument('--extract_stress', default=False, type=bool, help='Extract stress at each indentation step (will reduce simulation speed)')
    parser.add_argument('--export_results', default=True, type=bool, help='Export results to HDF')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--n_envs', default=1, type=int, help='Number of envs')
    
    args = parser.parse_args()

    # initialize gym
    gym = gymapi.acquire_gym()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Define sim parameters and create Sim object
    sim = create_sim(gym=gym,
                     gpu_physics=0,
                     gpu_render=0,
                     frict_coeff=args.frict_coeff)

    # Define and load assets
    asset_options = set_asset_options()
    #tube_urdf_dir = os.path.join('simulations', 'urdf', 'tubes')
    tube_urdf_dir = os.path.join('urdf', 'tubes')
    set_tube_matl_props(base_dir=tube_urdf_dir,
                        type=args.type,
                        resolution=args.resolution,
                        density=args.density,
                        elast_mod=args.elast_mod,
                        poiss_ratio=args.poiss_ratio)
    soft_asset = load_assets(gym=gym,
                             sim=sim,
                             base_dir=tube_urdf_dir,
                             object=args.type,
                             options=asset_options,
                             fix=False)
    
    # Define and create scene
    scene_props = set_scene_props(num_envs=args.n_envs)
    env_handles, actor_handles_tubes, up_rigid_body_handles, down_rigid_body_handles = create_scene(gym=gym, 
                                                                                                    sim=sim, 
                                                                                                    props=scene_props,
                                                                                                    assets_tube=soft_asset)    
    viewer, axes_geom = create_viewer(gym=gym, 
                                      sim=sim)
    
    
    # Run the main loop
    results = run_sim_loop(gym=gym, 
                           sim=sim, 
                           envs=env_handles, 
                           viewer=viewer,
                           actor_handles_tubes=actor_handles_tubes, 
                           up_rigid_body_handles=up_rigid_body_handles, 
                           down_rigid_body_handles=down_rigid_body_handles,
                           axes=axes_geom,
                           args=args)

    # Clean up
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
main()

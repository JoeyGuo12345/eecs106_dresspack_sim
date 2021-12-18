from ensenso_nxlib import NxLibCommand, NxLibException, NxLibItem
from ensenso_nxlib.constants import *
import ensenso_nxlib.api as api
import copy
import argparse
import open3d as o3d
import numpy as np

def gym_to_o3d(gym):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gym)
    return pcd

def get_camera_node(serial):
    root = NxLibItem()  # References the root
    cameras = root[ITM_CAMERAS][ITM_BY_SERIAL_NO]  # References the cameras subnode
    for i in range(cameras.count()):
        found = cameras[i].name() == serial
        if found:
            return cameras[i]

def _ensenso_to_open3d(ensenso_pc, b_box_0, b_box_1, b_box_2):
    point_cloud = o3d.geometry.PointCloud()

    # Reshape from (m x n x 3) to ( (m*n) x 3)
    vector_3d_vector = ensenso_pc.reshape(
        (ensenso_pc.shape[0] * ensenso_pc.shape[1]), ensenso_pc.shape[2])

    # Filter nans: if a row has nan's in it, delete it
    vector_3d_vector = vector_3d_vector[~np.isnan(
        vector_3d_vector).any(axis=1)]
    vector_3d_vector = vector_3d_vector[np.where(vector_3d_vector[:,1] >= b_box_0[0] - 30)]
    vector_3d_vector = vector_3d_vector[np.where(vector_3d_vector[:, 1] <= b_box_0[1] + 30)]
    vector_3d_vector = vector_3d_vector[np.where(vector_3d_vector[:,0] >= b_box_1[0] - 30)]
    vector_3d_vector = vector_3d_vector[np.where(vector_3d_vector[:, 0] <= b_box_1[1] + 30)]
    vector_3d_vector = vector_3d_vector[np.where(vector_3d_vector[:,2] >= b_box_2[0] - 30)]
    vector_3d_vector = vector_3d_vector[np.where(vector_3d_vector[:, 2] <= b_box_2[1] + 30)]
    # filtered_vector = np.array()
    # for point in vector_3d_vector:
    #     if
    point_cloud.points = o3d.utility.Vector3dVector(vector_3d_vector)
    return point_cloud
def combine_to_o3d(ensenso_pc, gym):
    pcd = o3d.geometry.PointCloud()
    # Reshape from (m x n x 3) to ( (m*n) x 3)
    vector_3d_vector = ensenso_pc.reshape(
        (ensenso_pc.shape[0] * ensenso_pc.shape[1]), ensenso_pc.shape[2])

    # Filter nans: if a row has nan's in it, delete it
    vector_3d_vector = vector_3d_vector[~np.isnan(
        vector_3d_vector).any(axis=1)]

    all_points = np.vstack((vector_3d_vector, gym))
    pcd.points = o3d.utility.Vector3dVector(all_points)
    return pcd

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--serial", type=str,
                        help="The serial of the stereo camera, you want to open")
    args = parser.parse_args(args)

    depth_camera_serial = '161080'
    rgb_camera_serial = '4103078743'

    gym = np.load('position1.npy', allow_pickle=True)[0]
    gym[:,1] -= 1.5

    gym = 1000*gym
    # Init Pose
    # prependicular to tube, + closer to me, - far from me
    gym[:, 1] += 315
    #z axis: - up, + down
    gym[:, 2] -= 900
    # parallel to tube, + to the right
    gym[:, 0] += 130

    # # prependicular to tube, + closer to me, - far from me
    # gym[:, 1] += 430
    # #z axis: - up, + down
    # gym[:, 2] -= 750
    # # parallel to tube, + to the right
    # gym[:, 0] += 90

    tube = gym_to_o3d(gym)

    tube_center = tube.get_center()
    tube_r = copy.deepcopy(tube)
    R = tube.get_rotation_matrix_from_xyz((0, np.pi / 2, -np.pi / 2))
    tube_r.rotate(R, center=tube_center)
    tube_r_np = np.asarray(tube_r.points)
    b_box_min_0 = np.min(tube_r_np[:, 1])
    b_box_max_0 = np.max(tube_r_np[:, 1])
    b_box_0 = (b_box_min_0, b_box_max_0)
    b_box_min_1 = np.min(tube_r_np[:, 0])
    b_box_max_1 = np.max(tube_r_np[:, 0])
    b_box_1 = (b_box_min_1, b_box_max_1)
    b_box_min_2 = np.min(tube_r_np[:, 2])
    b_box_max_2 = np.max(tube_r_np[:, 2])
    b_box_2 = (b_box_min_2, b_box_max_2)
    #o3d.visualization.draw_geometries([tube, tube_r])

    # Waits for the cameras to be initialized
    api.initialize()

    # Opens the camera with the serial stored in camera_serial variable
    cmd_depth = NxLibCommand(CMD_OPEN)
    cmd_depth.parameters()[ITM_CAMERAS] = depth_camera_serial
    cmd_depth.execute()

    # Captures with the previous openend depth camera
    capture_depth = NxLibCommand(CMD_CAPTURE)
    capture_depth.parameters()[ITM_CAMERAS] = depth_camera_serial
    capture_depth.execute()

    # Rectify the the captures raw images
    rectification_depth = NxLibCommand(CMD_RECTIFY_IMAGES)
    rectification_depth.execute()

    # Compute the disparity map
    disparity_map_depth = NxLibCommand(CMD_COMPUTE_DISPARITY_MAP)
    disparity_map_depth.execute()

    # Compute the point map from the disparitu map
    point_map_depth = NxLibCommand(CMD_COMPUTE_POINT_MAP)
    point_map_depth.execute()
    points = NxLibItem()[ITM_CAMERAS][depth_camera_serial][ITM_IMAGES][ITM_POINT_MAP].get_binary_data()
    #all_point = combine_to_o3d(points, gym)
    depth = _ensenso_to_open3d(points, b_box_0, b_box_1, b_box_2)
    dists = tube_r.compute_point_cloud_distance(depth)
    print(np.sum(dists)/len(dists))
    print(np.max(dists))

    o3d.visualization.draw_geometries([depth, tube_r])



    # Closes all open cameras
    NxLibCommand(CMD_CLOSE).execute()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

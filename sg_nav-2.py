# Modify the quantity of movement in default_structured_configs.py 
# (again to avoid following the same visual path)
# Include the option for colored pc and color-icp
# Consider initial alligment techniques (including AI)
# Try and error evaluation (since the complete fails...)
# I observe that it diverges when the wall appears on the image
# PB: Fails until 24 using SG and ICP
# Re implement ORB, BRISK and AKAZE
# Colored points instead of lines
# Inifite loop; not limited to 500 iterations

import os
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np
import open3d as o3d
import copy
import random
from omegaconf import OmegaConf
import argparse
import yaml
from scipy.spatial.transform import Rotation as R
import torch
from models.matching import Matching
from models.utils import frame2tensor

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

VM_PATH = "/home/rodriguez/Documents/logs/vm/"
LOGS_DIR = "/home/rodriguez/Documents/logs/"


class KeyBindings:
    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH_KEY = "f"
    MOVE_KEY = "m"
    RESET_VM_KEY = "k"
    NEXT_VM_KEY = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Run habitat simulation with options.")
    parser.add_argument("--save_all_imgs", action="store_true", help="Save all images.")
    parser.add_argument("--save_matched_points", action="store_true", help="Save matched points.")
    parser.add_argument("--save_pc", action="store_true", help="Save point cloud.")
    parser.add_argument("--visualize_registration", action="store_true", help="Visualize ICP registration.")
    parser.add_argument("--feature", type=str, default="ORB", choices=["ORB", "AKAZE", "BRISK", "SuperGlue"], 
                        help="Feature descriptor to be used for feature matching (ORB, AKAZE, BRISK, SuperGlue). Default is ORB.")
    parser.add_argument("--visual_path", action="store_true", help="Flag for visual path task only.")
    # Implement all the ICP variants
    return parser.parse_args()

def estimate_initial_alignment(source, target):
    """
    Estimate initial alignment using FPFH features and RANSAC.

    Args:
    source (open3d.geometry.PointCloud): Source point cloud.
    target (open3d.geometry.PointCloud): Target point cloud.

    Returns:
    numpy.ndarray: The estimated transformation matrix.
    """
    # Set parameters based on the point cloud statistics
    voxel_size = 5  # Adjusted voxel size
    radius_normal = voxel_size * 3  # Radius for normal estimation
    radius_feature = voxel_size * 5  # Radius for FPFH feature calculation
    max_correspondence_distance = 70 * 1.5  # Adjusted max correspondence distance

    # Downsample point clouds
    source_down = source.voxel_down_sample(voxel_size=voxel_size)
    target_down = target.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # RANSAC-based alignment
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=400000, confidence=0.999)
    )

    return result.transformation

def compute_normals(point_cloud, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    """
    Compute normals for a point cloud.

    Args:
    point_cloud (open3d.geometry.PointCloud): The point cloud for which normals are to be computed.
    search_param (open3d.geometry.KDTreeSearchParamHybrid): The parameters for KDTree search.

    Returns:
    open3d.geometry.PointCloud: The point cloud with normals computed.
    """
    point_cloud.estimate_normals(search_param=search_param)
    point_cloud.orient_normals_consistent_tangent_plane(50)  # Optional, for better normal orientation
    return point_cloud

def perform_icp(source, target, icp_variant='point_to_point', max_iteration=10000, distance_threshold=None, init_transform=np.eye(4)):
    """
    Perform ICP registration.

    Args:
    source (open3d.geometry.PointCloud): The source point cloud.
    target (open3d.geometry.PointCloud): The target point cloud.
    icp_variant (str): The variant of ICP to use. Options: 'point_to_point', 'point_to_plane', 'generalized'.
    max_iteration (int): Maximum number of iterations for ICP.
    distance_threshold (float): Distance threshold to consider for matching points.
    init_transform (np.array): Initial transformation guess.

    Returns:
    open3d.pipelines.registration.RegistrationResult: The result of ICP registration.
    """

    if distance_threshold is None:
        distance_threshold = np.linalg.norm(np.asarray(source.get_max_bound()) - np.asarray(source.get_min_bound())) * 0.05

    if icp_variant == 'point_to_point':
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        )
    elif icp_variant == 'point_to_plane':
        # Compute normals for the target point cloud
        target_with_normals = compute_normals(target)
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target_with_normals, distance_threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        )
    elif icp_variant == 'generalized':
        icp_result = o3d.pipelines.registration.registration_generalized_icp(
            source, target, distance_threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=2000,  # Max number of iterations
                relative_fitness=1e-6,  # Minimum relative improvement in fitness to continue
                relative_rmse=1e-6  # Minimum relative improvement in RMSE to continue
            )
        )
    else:
        raise ValueError("Unsupported ICP variant. Choose from 'point_to_point', 'point_to_plane', 'generalized'.")

    return icp_result

def determine_bot_action(T_result, verbose=True):
    """
    Determine the action a bot should take based on the transformation matrix.
    Args:
    T (np.array): A 4x4 transformation matrix containing rotation and translation.
    Returns:
    str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
    """
    # Extract the translation vector and Euler angles
    print('Processing action')
    T = np.copy(T_result)
    translation = T[0:3, 3]
    rotation_matrix = T[0:3, 0:3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    
    if verbose:
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

    # Define thresholds
    forward_threshold = 0.5 * 1.5  # meters
    lateral_threshold = 0.2  * 1.5 # meters
    yaw_threshold = 10 * 1.5      # degrees

    # Check translation for forward/backward movement
    if translation[0] < -forward_threshold:
        action_forward = 'Move Forward'
    else:
        action_forward = 'Stop'  # If the bot is close enough to the target

    # Check lateral translation and yaw angle for turning
    if translation[1] < -lateral_threshold or euler_angles[2] < -yaw_threshold:
        action_turn = 'Turn Right'
    elif translation[1] > lateral_threshold or euler_angles[2] > yaw_threshold:
        action_turn = 'Turn Left'
    else:
        action_turn = None  # No turn is needed if within thresholds

    # Combine actions: prioritize turning over moving forward
    if action_turn:
        return action_turn
    else:
        return action_forward

def handle_keystroke(keystroke, vm_image_index, VM_PATH, icp_result):
    if keystroke == ord(KeyBindings.MOVE_KEY):
        computed_action = 'Stop'
        if icp_result.fitness >= 0.1:  # Threshold for fitness, can be adjusted
            computed_action = determine_bot_action(icp_result.transformation)

        if computed_action == 'Move Forward':
            action = HabitatSimActions.move_forward
        elif computed_action == 'Turn Right':
            action = HabitatSimActions.turn_right
        elif computed_action == 'Turn Left':
            action = HabitatSimActions.turn_left
        elif computed_action == 'Stop':
            vm_image_index = (vm_image_index + 1) % len(os.listdir(VM_PATH + "color/"))
            display_visual_memory(VM_PATH, vm_image_index)
            return vm_image_index, None  # No action to execute

    elif keystroke == ord(KeyBindings.FORWARD_KEY):
        action = HabitatSimActions.move_forward

    elif keystroke == ord(KeyBindings.LEFT_KEY):
        action = HabitatSimActions.turn_left

    elif keystroke == ord(KeyBindings.RIGHT_KEY):
        action = HabitatSimActions.turn_right

    elif keystroke == ord(KeyBindings.FINISH_KEY):
        print("Finishing the episode.")
        return vm_image_index, "finish"  # Signal to finish the episode

    elif keystroke == ord(KeyBindings.RESET_VM_KEY):
        vm_image_index = 0
        display_visual_memory(VM_PATH, vm_image_index)
        return vm_image_index, None  # No action to execute

    elif keystroke == ord(KeyBindings.NEXT_VM_KEY):
        vm_image_index = (vm_image_index + 1) % len(os.listdir(VM_PATH + "color/"))
        display_visual_memory(VM_PATH, vm_image_index)
        return vm_image_index, None  # No action to execute

    else:
        return vm_image_index, None  # No action for unrecognized keystrokes

    # For actions that involve moving the agent
    return vm_image_index, action

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

K = np.array([
    [256, 0, 128],
    [0, 256, 128],
    [0, 0, 1]
])

def visualize_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imshow('Matched Features', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def combine_and_save_point_clouds(source, target, step_count):
    source_colored = copy.deepcopy(source)
    target_colored = copy.deepcopy(target)
    source_colored.paint_uniform_color([1, 0, 0])  # Red
    target_colored.paint_uniform_color([0, 0, 1])  # Blue
    combined_pcd = source_colored + target_colored
    combined_pcd_path = f"RGBD_sensor/log/combined_pc_step_{step_count:04d}.ply"
    o3d.io.write_point_cloud(combined_pcd_path, combined_pcd)
    print(f"Combined point cloud saved: {combined_pcd_path}")

def filter_invalid_points(points):
    valid_indices = np.where(points[:, 2] > 0)[0]
    return points[valid_indices]

def filter_invalid_points_with_color(points, colors):
    """
    Filter out points with invalid depth values and their corresponding colors.

    Args:
    points (np.ndarray): Array of points (shape Nx3).
    colors (np.ndarray): Array of colors corresponding to the points (shape Nx3).

    Returns:
    np.ndarray, np.ndarray: Filtered arrays of points and colors.
    """
    valid_indices = np.where(points[:, 2] > 0)[0]
    return points[valid_indices], colors[valid_indices]

def create_colored_point_cloud(points, colors):
    """
    Create an Open3D PointCloud object from points and colors.

    Args:
    points (np.ndarray): Array of points (shape Nx3).
    colors (np.ndarray): Array of colors corresponding to the points (shape Nx3).

    Returns:
    open3d.geometry.PointCloud: Colored point cloud in Open3D format.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def perform_colored_icp(source_pcd, target_pcd, voxel_size=2.5, max_iter=100, init_transform=np.eye(4)):
    """
    Perform Colored-ICP registration.

    Args:
    source_pcd (open3d.geometry.PointCloud): The source point cloud (colored).
    target_pcd (open3d.geometry.PointCloud): The target point cloud (colored).
    voxel_size (float): Voxel size for downsampling.
    max_iter (int): Maximum number of iterations for Colored-ICP.

    Returns:
    open3d.pipelines.registration.RegistrationResult: The result of the Colored-ICP.
    """

    # Downsample the point clouds
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=50))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=50))

    # Perform Colored-ICP
    result = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, max_correspondence_distance=voxel_size*1.5, init=init_transform,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    return result


def match_features(img1, img2, feature_descriptor='SuperGlue', device='cuda'):
    print(f'Using {feature_descriptor}')
    if feature_descriptor == 'ORB':
        # ORB is a good default choice
        detector = cv2.ORB_create()
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    elif feature_descriptor == 'AKAZE':
        # AKAZE features
        detector = cv2.AKAZE_create()
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    elif feature_descriptor == 'BRISK':
        # BRISK features
        detector = cv2.BRISK_create()
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    else:
        raise ValueError(f"Unsupported feature descriptor: {feature_descriptor}")

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_image = (1.0 - (image / np.max(image))) * 255.0
    depth_image = depth_image.astype(np.uint8)
    return depth_image

def save_images(rgb_image, depth_image, step_count):
    color_path = LOGS_DIR+f"all_images/color/{step_count:04d}.png"
    depth_path = LOGS_DIR+f"all_images/depth/{step_count:04d}.png"
    cv2.imwrite(color_path, rgb_image)
    cv2.imwrite(depth_path, depth_image)

def display_visual_memory(image_folder, current_image_index):
    color_image_folder = os.path.join(image_folder, "color/")
    depth_image_folder = os.path.join(image_folder, "depth/")

    color_image_files = sorted(os.listdir(color_image_folder))
    depth_image_files = sorted(os.listdir(depth_image_folder))

    if current_image_index < len(color_image_files) and current_image_index < len(depth_image_files):
        color_image_path = os.path.join(color_image_folder, color_image_files[current_image_index])
        depth_image_path = os.path.join(depth_image_folder, depth_image_files[current_image_index])

        color_image = cv2.imread(color_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        cv2.imshow("Target Color", color_image)
        cv2.imshow("Target Depth", depth_image)

def load_vm_images(VM_PATH, current_image_index):
    vm_path_color = VM_PATH + "color/"
    color_files = sorted(os.listdir(vm_path_color))
    if current_image_index < len(color_files):
        image_path = os.path.join(vm_path_color, color_files[current_image_index])
        color_vm_image = cv2.imread(image_path)

    vm_path_depth = VM_PATH + "depth/"
    depth_files = sorted(os.listdir(vm_path_depth))
    if current_image_index < len(depth_files):
        image_path = os.path.join(vm_path_depth, depth_files[current_image_index])
        depth_vm_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    return color_vm_image, depth_vm_image

def get_3d_points(kp1, kp2, matches, depth_img1, depth_img2, K):
    points1 = []
    points2 = []
    
    for m in matches:
        # Image 1
        u1, v1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
        z1 = depth_img1[v1, u1]
        if isinstance(z1, np.ndarray):
            z1 = z1[0] if z1.shape[0] > 0 else None

        # Image 2
        u2, v2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])
        z2 = depth_img2[v2, u2]
        if isinstance(z2, np.ndarray):
            z2 = z2[0] if z2.shape[0] > 0 else None

        if z1 is not None and z1 > 0 and z2 is not None and z2 > 0:
            x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
            y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
            points1.append([x1, y1, z1])

            x2 = (u2 - K[0, 2]) * z2 / K[0, 0]
            y2 = (v2 - K[1, 2]) * z2 / K[1, 1]
            points2.append([x2, y2, z2])

    return np.array(points1), np.array(points2)

def sg_get_3d_points(kp1, kp2, depth_img1, depth_img2, K):
    points1 = []
    points2 = []
    len_matches = len(kp1)
    for m in range(len_matches):
        # Image 1
        u1, v1 = int(kp1[m][0]), int(kp1[m][1])
        z1 = depth_img1[v1, u1]
        if isinstance(z1, np.ndarray):
            z1 = z1[0] if z1.shape[0] > 0 else None

        # Image 2
        u2, v2 = int(kp2[m][0]), int(kp2[m][1])
        z2 = depth_img2[v2, u2]
        if isinstance(z2, np.ndarray):
            z2 = z2[0] if z2.shape[0] > 0 else None

        if z1 is not None and z1 > 0 and z2 is not None and z2 > 0:
            x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
            y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
            points1.append([x1, y1, z1])

            x2 = (u2 - K[0, 2]) * z2 / K[0, 0]
            y2 = (v2 - K[1, 2]) * z2 / K[1, 1]
            points2.append([x2, y2, z2])

    return np.array(points1), np.array(points2)

def sg_get_3d_points_with_color(kp1, kp2, depth_img1, depth_img2, color_img1, color_img2, K):
    points1 = []
    colors1 = []
    points2 = []
    colors2 = []
    len_matches = len(kp1)

    for m in range(len_matches):
        # Image 1
        u1, v1 = int(kp1[m][0]), int(kp1[m][1])
        z1 = depth_img1[v1, u1]
        if isinstance(z1, np.ndarray):
            z1 = z1[0] if z1.shape[0] > 0 else None
        color1 = color_img1[v1, u1] / 255.0  # Normalize color to [0, 1]

        # Image 2
        u2, v2 = int(kp2[m][0]), int(kp2[m][1])
        z2 = depth_img2[v2, u2]
        if isinstance(z2, np.ndarray):
            z2 = z2[0] if z2.shape[0] > 0 else None
        color2 = color_img2[v2, u2] / 255.0  # Normalize color to [0, 1]

        if z1 is not None and z1 > 0 and z2 is not None and z2 > 0:
            x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
            y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
            points1.append([x1, y1, z1])
            colors1.append(color1)

            x2 = (u2 - K[0, 2]) * z2 / K[0, 0]
            y2 = (v2 - K[1, 2]) * z2 / K[1, 1]
            points2.append([x2, y2, z2])
            colors2.append(color2)

    return np.array(points1), np.array(colors1), np.array(points2), np.array(colors2)


def draw_matches(img1, mkpts0, img2, mkpts1):
    """ Draw matches on the images. """
    # Create a blank image that fits both the input images
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    matched_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place the first image on the left
    matched_img[:img1.shape[0], :img1.shape[1]] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    # Place the next image to the right of the first image
    matched_img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw lines between matched keypoints
    for pt1, pt2 in zip(mkpts0, mkpts1):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + img1.shape[1]), int(pt2[1]))
        cv2.line(matched_img, pt1, pt2, (255, 0, 0), 1)
    
    return matched_img

def sg_draw_matches(image1, image2, mkpts0, mkpts1, scale=1):
    """ Draw lines connecting matched keypoints between image1 and image2. """
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    matched_image = np.zeros((height, width, 3), dtype=np.uint8)
    matched_image[:h1, :w1] = image1
    matched_image[:h2, w1:w1+w2] = image2

    for pt1, pt2 in zip(mkpts0, mkpts1):
        pt1 = (int(round(pt1[0])), int(round(pt1[1])))
        pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

        cv2.circle(matched_image, pt1, 3, (0, 255, 0), -1)
        cv2.circle(matched_image, pt2, 3, (0, 255, 0), -1)
        cv2.line(matched_image, pt1,pt2, (255, 0, 0), 1)

    # Resize for visualization
    matched_image = cv2.resize(matched_image, (width // scale, height // scale))

    return matched_image


def process_and_match_images(image1, image2, device):
    g_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    g_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    frame_tensor1 = frame2tensor(g_image1, device)
    frame_tensor2 = frame2tensor(g_image2, device)
    
    matching = Matching({'superpoint': {}, 'superglue': {'weights': 'indoor'}}).to(device).eval()
    with torch.no_grad():  # No need to track gradients
        pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})
    
    # Detach tensors before converting to NumPy arrays
    kpts0 = pred['keypoints0'][0].cpu().detach().numpy()
    kpts1 = pred['keypoints1'][0].cpu().detach().numpy()
    matches = pred['matches0'][0].cpu().detach().numpy()
    confidence = pred['matching_scores0'][0].cpu().detach().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    conf = confidence[valid]

    return mkpts0, mkpts1, conf

def select_high_confidence_points(mkp1, mkp2, confidences, threshold):
    kp1 = []
    kp2 = []

    for i, confidence in enumerate(confidences):
        if confidence > threshold:
            kp1.append(mkp1[i])
            kp2.append(mkp2[i])

    return np.asarray(kp1), np.asarray(kp2)

def demo():
    # Define the device for computation at the start of the demo function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    env = habitat.Env(config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml"))

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    print("Agent stepping around inside environment.")
    
    count_steps = 0
    vm_image_index = 0
    current_color = transform_rgb_bgr(observations["rgb"])
    current_depth = transform_depth(observations["depth"])
    cv2.imshow("Current Color", current_color)
    cv2.imshow("Current Depth", current_depth)
    display_visual_memory(VM_PATH, vm_image_index)
    th_conf = 0.5
    colored_icp = False

    while not env.episode_over:
        target_color, target_depth = load_vm_images(VM_PATH, vm_image_index)

        if args.save_all_imgs or args.visual_path:
            save_images(current_color, current_depth, count_steps)

        if args.visual_path:
            icp_result = 0
        else:
            # Assume SuperGlue is used for simplicity; adapt as needed for other methods
            print("Using SuperGlue (only this one is currently implemented).")
            mkp1, mkp2, confidences = process_and_match_images(current_color, target_color, device)
            print(f"Found {len(confidences)} matches.")
            kp1, kp2 = select_high_confidence_points(mkp1, mkp2, confidences, th_conf)
            print("Number of trusted matched points: ", len(kp1))

            if args.save_matched_points and len(kp1) > 0:
                matched_img = sg_draw_matches(current_color, target_color, kp1, kp2)
                match_img_path = os.path.join(LOGS_DIR, f"matched_points/match_{vm_image_index:04d}_{len(mkp1)}.png")
                cv2.imwrite(match_img_path, matched_img)
            
            if colored_icp:
                points1, colors1, points2, colors2 = sg_get_3d_points_with_color(kp1, kp2, current_depth, target_depth, current_color, target_color, K)
                filtered_points1, filtered_colors1 = filter_invalid_points_with_color(points1, colors1)
                filtered_points2, filtered_colors2 = filter_invalid_points_with_color(points2, colors2)
                pc1 = create_colored_point_cloud(filtered_points1, filtered_colors1)
                pc2 = create_colored_point_cloud(filtered_points2, filtered_colors2)
            else:
                points1, points2 = sg_get_3d_points(kp1, kp2, current_depth, target_depth, K)
                points1 = filter_invalid_points(points1.reshape(-1, 3))
                points2 = filter_invalid_points(points2.reshape(-1, 3))
                pc1 = o3d.geometry.PointCloud()
                pc1.points = o3d.utility.Vector3dVector(points1)
                pc2 = o3d.geometry.PointCloud()
                pc2.points = o3d.utility.Vector3dVector(points2)

            # Save point clouds if needed
            if args.save_pc and len(points1) > 0 and len(points2) > 0:
                pc_dir = LOGS_DIR + 'point_clouds'
                os.makedirs(pc_dir, exist_ok=True)
                pc1_path = pc_dir + f"/pc_current_{count_steps:04d}.ply"
                pc2_path = pc_dir + f"/pc_target_{vm_image_index:04d}.ply"
                o3d.io.write_point_cloud(pc1_path, pc1)
                o3d.io.write_point_cloud(pc2_path, pc2)

            # Initial alignment estimation
            initial_transformation = estimate_initial_alignment(pc1, pc2)

            # Perform ICP registration
            if colored_icp:
                icp_result = perform_colored_icp(pc1, pc2, voxel_size=2.5, max_iter=10000, init_transform=initial_transformation)
            else:
                icp_result = perform_icp(pc1, pc2, icp_variant='generalized', max_iteration=10000, init_transform=initial_transformation)

            print("ICP Registration result:")
            print(icp_result.transformation)
            
            th_fit = 0.1
            computed_action = determine_bot_action(icp_result.transformation) if icp_result.fitness >= th_fit else 'Stop'
            print("Computed action: ", computed_action)
            
            # Check fitness of ICP registration
            if icp_result.fitness < th_fit:  # Adjust the fitness threshold as needed
                print("ICP registration failed.")
            else:
                print("Trusted action.")

            if args.visualize_registration:
                # Optionally visualize the registration result
                pc1.transform(icp_result.transformation)
                o3d.visualization.draw_geometries([pc1, pc2])
        print('Step: ', count_steps)
        keystroke = cv2.waitKey(0)
        vm_image_index, action = handle_keystroke(keystroke, vm_image_index, VM_PATH, icp_result)

        if action == "finish":
            break
        elif action:
            observations = env.step(action)
            # Update the current_color and current_depth after the action
            current_color = transform_rgb_bgr(observations["rgb"])
            current_depth = transform_depth(observations["depth"])
            count_steps += 1
            cv2.imshow("Current Color", current_color)
            cv2.imshow("Current Depth", current_depth)

    print("Episode finished after {} steps.".format(count_steps))

if __name__ == "__main__":
    args = parse_args()
    demo()

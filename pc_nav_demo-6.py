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

from models.superpoint import SuperPoint
from models.superglue import SuperGlue

# Initialize SuperPoint with default configuration
superpoint = SuperPoint(SuperPoint.default_config)
# Initialize SuperGlue with default configuration
superglue = SuperGlue(SuperGlue.default_config)

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

VM_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/RGBD_sensor/vm/"
LOGS_DIR = "/home/rodriguez/Documents/logs/"

def parse_args():
    parser = argparse.ArgumentParser(description="Run habitat simulation with options.")
    parser.add_argument("--save_all_imgs", action="store_true", help="Save all images.")
    parser.add_argument("--save_matched_points", action="store_true", help="Save matched points.")
    parser.add_argument("--save_pc", action="store_true", help="Save point cloud.")
    parser.add_argument("--visualize_registration", action="store_true", help="Visualize ICP registration.")

    return parser.parse_args()

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
    forward_threshold = 0.5  # meters
    lateral_threshold = 0.2  # meters
    yaw_threshold = 10       # degrees

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

def preprocess_image(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to float and normalize
    img = img.astype('float32') / 255.0

    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img_tensor

def match_features(img1, img2, feature_descriptor='SuperGlue'):
    if feature_descriptor in ['ORB', 'AKAZE', 'BRISK']:
        # Initialize the detector for the chosen feature descriptor
        if feature_descriptor == 'ORB':
            detector = cv2.ORB_create()
        elif feature_descriptor == 'AKAZE':
            detector = cv2.AKAZE_create()
        else:  # BRISK
            detector = cv2.BRISK_create()

        # Detect and compute features
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        # Match features using BFMatcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return kp1, kp2, matches

    elif feature_descriptor == 'SuperGlue':
        img1_tensor = preprocess_image(img1)
        img2_tensor = preprocess_image(img2)
        # Detect and describe features using SuperPoint
        sp_data1 = {'image': img1_tensor}
        sp_data2 = {'image': img2_tensor}
        sp_out1 = superpoint(sp_data1)
        sp_out2 = superpoint(sp_data2)

        # Initialize keypoints
        kpts0 = torch.empty((0, 2))
        kpts1 = torch.empty((0, 2))

        # Check if keypoints are detected and handle them
        if len(sp_out1['keypoints']) > 0:
            kpts0 = sp_out1['keypoints'][0]
            height, width = img1_tensor.shape[2], img1_tensor.shape[3]
            kpts0 = kpts0 / torch.tensor([width, height], device=kpts0.device)

        if len(sp_out2['keypoints']) > 0:
            kpts1 = sp_out2['keypoints'][0]
            height, width = img2_tensor.shape[2], img2_tensor.shape[3]
            kpts1 = kpts1 / torch.tensor([width, height], device=kpts1.device)

        # Ensure scores are in the correct shape
        scores0 = sp_out1['scores'][0].unsqueeze(1) if len(sp_out1['scores']) > 0 else torch.empty((0, 1))
        scores1 = sp_out2['scores'][0].unsqueeze(1) if len(sp_out2['scores']) > 0 else torch.empty((0, 1))

        # Prepare data for SuperGlue
        sg_data = {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': sp_out1['descriptors'][0],
            'descriptors1': sp_out2['descriptors'][0],
            'scores0': scores0,
            'scores1': scores1,
            'image0': img1_tensor,
            'image1': img2_tensor,
        }

        # Match features using SuperGlue
        sg_output = superglue(sg_data)
        matches = sg_output['matches0']  # Matches from image1 to image2

        # Convert matches to the format you need
        # ...

        return kpts0, kpts1, matches

    else:
        raise ValueError(f"Unsupported feature descriptor: {feature_descriptor}")

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


def demo():
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

    while not env.episode_over:

        target_color, target_depth = load_vm_images(VM_PATH, vm_image_index)
        kp1, kp2, matches = match_features(current_color, target_color)
        print(f"Feature matching done with {len(matches)} matched points.")

        if args.save_matched_points:
            matched_img = cv2.drawMatches(current_color, kp1, target_color, kp2, matches, None, flags=2)
            match_img_path = LOGS_DIR+f"matched_points/match_{vm_image_index:04d}_{len(matches)}.png"
            cv2.imwrite(match_img_path, matched_img)

        points1, points2 = get_3d_points(kp1, kp2, matches, current_depth, target_depth, K)
        points1 = filter_invalid_points(points1.reshape(-1, 3))
        points2 = filter_invalid_points(points2.reshape(-1, 3))

        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(points1)
        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(points2)

        if args.save_pc and len(points1) > 0 and len(points2) > 0:
            pc_dir = LOGS_DIR + 'point_clouds'
            os.makedirs(pc_dir, exist_ok=True)
            pc1_path = pc_dir + f"/pc_current_{count_steps:04d}.ply"
            pc2_path = pc_dir + f"/pc_target_{vm_image_index:04d}.ply"
            o3d.io.write_point_cloud(pc1_path, pc1)
            o3d.io.write_point_cloud(pc2_path, pc2)

        # Point Cloud Registration using ICP
        icp_threshold = np.linalg.norm(np.asarray(pc1.get_max_bound()) - np.asarray(pc1.get_min_bound())) * 0.05
        icp_result = o3d.pipelines.registration.registration_icp(
            pc1, pc2, icp_threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

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

        if args.save_all_imgs:
            save_images(current_color, current_depth, count_steps)
        
        keystroke = cv2.waitKey(0)

        if keystroke == ord('m'):
            if computed_action == 'Move Forward':
                action = HabitatSimActions.move_forward
                observations = env.step(action)
            elif computed_action == 'Turn Right':
                action = HabitatSimActions.turn_right
                observations = env.step(action)
            elif computed_action == 'Turn Left':
                action = HabitatSimActions.turn_left
                observations = env.step(action)
            elif computed_action == 'Stop':
                # Update visual memory only if the computed action is 'Stop'
                vm_image_index = (vm_image_index + 1) % len(os.listdir(VM_PATH + "color/"))
                display_visual_memory(VM_PATH, vm_image_index)
                # No action to execute, so no env.step() call here

        elif keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            observations = env.step(action)

        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            observations = env.step(action)

        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            observations = env.step(action)
        
        elif keystroke == ord(FINISH):
            print("Finishing the episode.")
            break  # This exits the while loop, effectively ending the episode

        elif keystroke == ord('k'):
            vm_image_index = 0
            display_visual_memory(VM_PATH, vm_image_index)

        elif keystroke == ord('0'):
            vm_image_index = (vm_image_index + 1) % len(os.listdir(VM_PATH + "color/"))
            display_visual_memory(VM_PATH, vm_image_index)

        current_color = transform_rgb_bgr(observations["rgb"])
        current_depth = transform_depth(observations["depth"])
        count_steps += 1

        cv2.imshow("Current Color", current_color)
        cv2.imshow("Current Depth", current_depth)

    print("Episode finished after {} steps.".format(count_steps))

if __name__ == "__main__":
    args = parse_args()
    demo()

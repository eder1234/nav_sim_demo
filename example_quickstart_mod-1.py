import os
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np
from omegaconf import OmegaConf

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

SOURCE_DIR = "RGBD_sensor/"

def create_directories():
    os.makedirs(SOURCE_DIR+"color", exist_ok=True)
    os.makedirs(SOURCE_DIR+"depth", exist_ok=True)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    # Normalize depth to a grayscale image
    depth_image = (1.0 - (image / np.max(image))) * 255.0
    depth_image = depth_image.astype(np.uint8)
    return depth_image


def save_images(rgb_image, depth_image, step_count):
    color_path = f"RGBD_sensor/color/{step_count:04d}.png"
    depth_path = f"RGBD_sensor/depth/{step_count:04d}.png"
    cv2.imwrite(color_path, rgb_image)
    cv2.imwrite(depth_path, depth_image)

def example():
    create_directories()
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    config_dict = OmegaConf.to_container(config, resolve=True)
    modified_config = OmegaConf.create(config_dict)
    env = habitat.Env(config=modified_config)

    observations = env.reset()
    rgb_image = transform_rgb_bgr(observations["rgb"])
    depth_image = transform_depth(observations["depth"])

    count_steps = 0
    while not env.episode_over:
        cv2.imshow("RGB", rgb_image)
        cv2.imshow("Depth", depth_image)
        keystroke = cv2.waitKey(0)

        # Save images on each step
        save_images(rgb_image, depth_image, count_steps)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        rgb_image = transform_rgb_bgr(observations["rgb"])
        depth_image = transform_depth(observations["depth"])
        count_steps += 1

    print("Episode finished after {} steps.".format(count_steps))

if __name__ == "__main__":
    example()

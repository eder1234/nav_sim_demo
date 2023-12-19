import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np
from omegaconf import OmegaConf

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    # Normalize depth for display
    depth_image = (image / np.max(image) * 255).astype(np.uint8)
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return depth_image

def example():
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

    # Convert to a native Python dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Print the configuration structure for inspection
    print("Config structure:", config_dict)

    # Find and modify the configuration for sensors
    # Placeholder for the actual modification
    # config_dict["CORRECT_PATH_TO_SENSORS"] = ["RGB_SENSOR", "DEPTH_SENSOR"]

    # Convert back to an OmegaConf object
    modified_config = OmegaConf.create(config_dict)

    # Initialize the environment with the modified configuration
    env = habitat.Env(config=modified_config)
    
    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    cv2.imshow("Depth", transform_depth(observations["depth"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("Depth", transform_depth(observations["depth"]))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")

if __name__ == "__main__":
    example()

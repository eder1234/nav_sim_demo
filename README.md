# PC Nav Demo-4

This Python script, `pc_nav_demo-4.py`, is designed for navigating a robot in a virtual environment using the Habitat simulator. It employs computer vision techniques for processing and matching images, as well as handling point clouds for navigation.

## Getting Started

Before running the script, make sure to activate the Conda environment with the necessary dependencies:

```bash
conda activate habitat
```

## Command Line Arguments

Run the script with the following command line arguments to enable specific functionalities:

- `--save_all_imgs`: Save all images captured during the simulation.
- `--save_matched_points`: Save images of matched points between pairs of images.
- `--save_pc`: Save point clouds generated during the simulation.
- `--visualize_registration`: Visualize the ICP registration process.

Example:

```bash
python pc_nav_demo-4.py --save_all_imgs --save_matched_points
```

## Keyboard Controls

During the simulation, the following keyboard keys can be used for various actions:

- **0**: Update visual memory image index. This cycles through the saved images in the VM_PATH directory.
- **p**: Process the current view with the target view based on the visual memory index. It performs feature matching, point cloud generation, and ICP registration.
- **w (FORWARD_KEY)**: Move the robot forward.
- **a (LEFT_KEY)**: Turn the robot left.
- **d (RIGHT_KEY)**: Turn the robot right.
- **f (FINISH)**: Stop the robot and end the simulation.

## Expected Behavior

- Pressing **0** cycles through the stored images in the visual memory, updating the current target view.
- Pressing **p** initiates processing of the current and target views. It matches features, generates point clouds, and performs ICP registration. The action for the robot (move forward, turn, stop) is determined based on this processing.
- The **w, a, d** keys control the robot's movement within the environment, allowing for manual navigation.
- Pressing **f** will stop the simulation and exit the program.

## Additional Notes

- Ensure the VM_PATH and LOGS_DIR directories are correctly set for storing and accessing images and logs.
- Adjust the script parameters and thresholds according to the specifics of your simulation environment and robot capabilities.

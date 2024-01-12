# PC Navigation Demo (pc_nav_demo-10.py)

This repository contains the `pc_nav_demo-10.py` script, a Python program designed for demonstrating navigation and feature matching in a point cloud (PC) environment. The script is part of a project that utilizes Habitat Lab and other computer vision libraries.

## Prerequisites

Before running the script, ensure you have the following prerequisites:

- Python 3.9
- Habitat Lab
- OpenCV
- Open3D
- NumPy
- SciPy
- PyTorch
- Argparse
- YAML

## Environment Setup

To run the script, you need to set up a conda environment with Habitat Lab. Follow these steps:

1. Install [Anaconda](https://www.anaconda.com/products/individual) or Miniconda.
2. Create a new conda environment as in habitat-lab:

    ```bash
    conda create -n habitat python=3.9
    ```

3. Activate the conda environment:

    ```bash
    conda activate habitat
    ```

4. Install Habitat Lab and other required packages in your environment. Refer to the [Habitat Lab documentation](https://aihabitat.org/docs/habitat-lab/) for detailed installation instructions.

## Running the Script

After setting up and activating the conda environment, you can run the script as follows:

```bash
python pc_nav_demo-10.py [arguments]
```

### Script Arguments

The script accepts several command-line arguments to customize its behavior:

- `--save_all_imgs`: Save all images during the navigation.
- `--save_matched_points`: Save matched points between successive images.
- `--save_pc`: Save the generated point clouds.
- `--visualize_registration`: Visualize the ICP registration process.
- `--feature`: Specify the feature descriptor to be used (e.g., ORB, AKAZE, BRISK, SuperGlue).

Example:

```bash
python pc_nav_demo-10.py --save_all_imgs --feature ORB
```

## Key Bindings

During the execution, the script responds to the following key bindings:

- `w`: Move forward.
- `a`: Turn left.
- `d`: Turn right.
- `f`: Finish the episode.
- `m`: Automatic movement based on computed action.
- `k`: Reset the visual memory index.
- `0`: Advance to the next visual memory image.

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository, make your changes, and create a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).

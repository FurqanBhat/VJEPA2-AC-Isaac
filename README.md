# V-JEPA 2-AC Isaac Sim Planner

This repository contains a closed-loop Model Predictive Control (MPC) planner for a Franka Emika Panda arm in NVIDIA Isaac Sim. It leverages the [V-JEPA 2](https://github.com/facebookresearch/vjepa2) action-conditioned (AC) predictor for goal-image directed manipulation via ROS 2.

## Overview

The system operates using a single goal image—no explicit trajectories or gripper routines are scripted. The control loop continuously:
1. **Encodes** the current RGB frame into a latent representation.
2. **Plans** using the Cross-Entropy Method (CEM) over the AC predictor to find the sequence of actions that minimizes the L1 distance to the goal image latent.
3. **Executes** the first action using a local damped-least-squares Inverse Kinematics (IK) solver.
4. **Waits** for the arm to settle and a fresh frame to arrive before replanning.

## Prerequisites

- **NVIDIA Isaac Sim** (with ROS 2 extension enabled)
- **ROS 2** (Humble or newer)
- **Python 3.10+** (Matching your Isaac Sim environment)

### Python Dependencies

Install the required Python packages in your environment:

```bash
pip install torch torchvision numpy opencv-python scipy roboticstoolbox-python spatialmath-python

```

*(Note: `rclpy`, `sensor_msgs`, and `cv_bridge` must be available via your ROS 2 environment sourcing).*

## Model Checkpoint Setup

The script expects the V-JEPA 2-AC ViT-Giant checkpoint. By default, `torch.hub` looks in its local cache. You must download the official checkpoint and point the script to it.

The script will automatically symlink your local checkpoint into the `torch.hub` cache directory. Ensure the `ckpt` parameter points to your downloaded weights (default: `/home/ediilebudu/vjepa2-ac-vitg.pt`).

## ROS 2 Interface

The node (`vjepa2_ac_mpc`) communicates with Isaac Sim via the following topics:

### Subscriptions

* **`/rgb`** (`sensor_msgs/Image`): The current camera observation.
* **`/joint_states`** (`sensor_msgs/JointState`): Current joint positions and velocities of the robot.

### Publishers

* **`/joint_command`** (`sensor_msgs/JointState`): Target joint positions published to the Isaac Sim articulation controller.

## Usage

1. Launch your Isaac Sim environment with the ROS 2 bridge enabled and the Franka robot configured to publish/subscribe to the topics listed above.
2. Source your ROS 2 workspace:
```bash
source /opt/ros/humble/setup.bash

```


3. Run the planner script, providing the path to your goal image:

```bash
python3 scripts/run_vjepa2.py --ros-args -p goal_image:=goal_image.png -p max_steps:=40

```

## Key Parameters

You can tune the behavior of the planner at runtime using ROS 2 parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `goal_image` | `goal_image.png` | Path to the target RGB image. |
| `ckpt` | `/home/ediilebudu/vjepa2-ac-vitg.pt` | Path to the downloaded V-JEPA 2 weights. |
| `precision` | `fp32` | Compute precision (`fp32` or `bf16`). |
| `max_steps` | `30` | Maximum number of MPC steps before exiting. |
| `action_yaw_deg` | `-120.0` | Calibration angle between the camera frame and robot base frame. |
| `horizon` | `1` | MPC prediction horizon. |
| `num_samples` | `256` | Number of action sequences sampled per CEM iteration. |
| `cem_iters` | `2` | Number of Cross-Entropy Method iterations. |
| `max_action_m` | `0.075` | Maximum translational step size (meters). |
| `gripper_type` | `robotiq` | Gripper configuration (`robotiq` or `franka`). |



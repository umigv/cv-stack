
# drivable_area

This package generates and publishes an occupancy grid based on a ZED camera image and a bird's eye view transform. It uses OpenCV for image processing and ROS2 (rclpy) for communication.

## Installation

1. Clone this repository into your ROS2 workspace (`/root/ros2_ws/src`):
   ```
   cd /root/ros2_ws/src
   git clone <repository-url> drivable_area
   ```
2. Build the workspace:
   ```
   cd /root/ros2_ws
   colcon build --symlink-install
   ```
3. Source your workspace:
   ```
   source /root/ros2_ws/install/setup.bash
   ```

## Package Dependencies

- rclpy
- sensor_msgs
- std_msgs
- nav_msgs
- cv_bridge
- OpenCV
- numpy
- matplotlib (for testing subscriber)
- skimage

## Configuration

All configuration parameters (camera settings, image dimensions, occupancy grid offsets, HSV filter settings, etc.) can be modified using launch file arguments. For example, you can launch the node with custom configuration:

```
ros2 launch drivable_area drivable_area.launch.py \
 image.width:=1920 image.height:=1080 hsv.lower:="[10, 20, 140]" hsv.upper:="[170, 50, 255]" morph.iterations:=3
```

## Usage

### Launching the Node

To start the drivable_area node, run the launch file:

```
ros2 launch drivable_area drivable_area.launch.py
```

This will start your processing node (drivable_area) with the default or overridden parameters.

### Testing

A test subscriber (`drivable_area_test_subscriber.py`) and test publisher (`drivable_area_test_publisher.py`) are provided.

- To test publishing, run:
  ```
  ros2 run drivable_area drivable_area_test_publisher
  ```
- In another terminal, run the subscriber:
  ```
  ros2 run drivable_area drivable_area_test_subscriber
  ```

The subscriber will log occupancy grid information and save an image (`occupancy_grid.png`) showing the occupancy result.
The publisher will simulate a ZED video stream from the test video provided in the python script.

## Files

- `/drivable_area/drivable_area/drivable_area.py`: Main node that processes incoming images to generate an occupancy grid.
- `/drivable_area/launch/drivable_area.launch.py`: Launch file with configurable parameters.
- `/drivable_area/drivable_area/drivable_area_test_subscriber.py`: Test subscriber to inspect occupancy grid messages.
- `/drivable_area/drivable_area/drivable_area_test_publisher.py`: Test publisher that simulates a ZED image stream.

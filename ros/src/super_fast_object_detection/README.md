# ROS Package for Super Fast Accurate 3D object detection


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)  
Ubuntu 18.04 & ROS Melodic

## Installation

Clone and setup the main python package
```
git clone https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection
cd Super-Fast-Accurate-3D-Object-Detection/
pip install .
```
Install dependancies for ROS packages:
```
sudo apt install ros-melodic-autoware-msgs
```
## Building Workspace
```
cd Super-Fast-Accurate-3D-Object-Detection/ros/
catkin_make
```

## Running the node
Run the node by simply after you build the workspace
```
source devel/setup.bash
rosrun super_fast_object_detection rosInference.py
```

### Subscriber
Topic Name: ```points_raw```, Message Type: ```sensor_msgs/PointCloud2```
### Publisher
Topic Name: ```detected_objects```, Message Type: ```autoware_msgs/DetectedObjectArray```

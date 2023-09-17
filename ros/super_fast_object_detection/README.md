# ROS Package for Super Fast Accurate 3D object detection


Install dependancies for ROS packages:
```
sudo apt install ros-melodic-autoware-msgs
```
## Building Workspace
```
cd Super-Fast-Accurate-3D-Object-Detection/ros/
catkin_make -DCMAKE_BUILD_TYPE=Release
```

## Running the node
Run the node by simply after you build the workspace
```
source devel/setup.bash
rosrun super_fast_object_detection rosInference.py
```

Note: If you want to visualize that detected objects, you can use detected_objects_visualizer package.

### Subscriber
Topic Name: ```points_raw```, Message Type: ```sensor_msgs/PointCloud2```
### Publisher
Topic Name: ```detected_objects```, Message Type: ```autoware_msgs/DetectedObjectArray```

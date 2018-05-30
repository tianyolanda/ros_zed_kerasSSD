# ros_zed_kerasSSD
A ros package for autonomous driving object detection.: ZED camera, ROS message, A keras implemented SSD 

# detect_pkg
ros_zed_kerasSSD

A ros package for autonomous driving object detection.

It contains modules of:
1. A keras implemented SSD
2. Subscribe from ZED Camera ROS node, obtained the PointCloud information of Object bounding box center
3. Publish a ROS message about the object information(object type/confidence/distance/)

## Environment
- Ubuntu16.04
- CUDA 9.0
- CUDNN 7.0.5
- ROS Kinetic
- ZED Camera SDK 2.2.1
- zed-ros-wrapper-2.2.x
- Python version 3.5
- tensorflow-gpu (1.5.0rc1)
- Keras (2.1.3)
- cv-bridge (1.12.7)


## pre-install
### Environment
### others:
- cv_bridge for python3

## Usage
clone and unziped into your ROS workspace/src(suppose workspace names catkin_ws)


```
# for the first time
cd ~/catkin_ws 
catkin_make -DCATKIN_WHITELIST_PACKAGES=detect_pkg
cd src/detect_pkg/
chmod u+x bin/hello
cd ~/catkin_ws
. devel/setup.bash

# run it!!
roslaunch zed_wrapper zed.launch
rosrun detect_pkg hello
```
Note that:
hello
hello_ssd300
hello_512
are the program entrance. Each of them can rosrun after make them executable.

## weight file download
https://drive.google.com/open?id=1mVDBgZ4u-6BS8GMYeR4j3UUpuHaMkiVe
download the needed weight file and save it in detect_pkg/bin/

## Performance
GTX GFORCE 1070 

ssd300: 13fps
ssd512: 8fps



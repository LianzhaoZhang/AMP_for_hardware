cmake_minimum_required(VERSION 3.10)
project(deploy_mujoco)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    rospy
)

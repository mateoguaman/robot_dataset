#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import jax 
import jax.numpy as jnp
import time
import gymnasium as gym
from gymnasium.spaces import Space, Dict, Discrete, Box

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from cv_bridge import CvBridge

from robot_dataset.data.replay_buffer import ReplayBuffer
from robot_dataset.config_parser.config_parser import ConfigParser
from robot_dataset.online_converter.robot_listener import RobotListener

model = 
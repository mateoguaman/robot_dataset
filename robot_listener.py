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
from gym.spaces import Space, Dict, Discrete, Box, Sequence

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from cv_bridge import CvBridge

from robot_dataset.data.replay_buffer import ReplayBuffer

def image_processor(image):
    '''Converts image to Gym Space'''
    pass

def imu_processor(imu_sequence):
    '''Converts sequence of IMU data to Gym Space'''
    pass

class Topic(object):
    def __init__(self, topic_name, msg_type, io_function, action=False):
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.io_function = io_function
        self.action = action

    def process(self, msg):
        processed_msg = self.io_function(msg)
        return processed_msg

def _obs_act_space_from_topics(topics):
    '''
        Args:
            - topics: List of Topic objects
        Returns:
            - obs_space: gym.Spaces space (most likely Dict) for all topics in list
            - action_space: gym.Spaces space for action space, could be empty if no action in topics
    '''
    pass

class RobotListener(object):
    def __init__(self, topics, buffer_capacity=10000):
        '''
        Args:
            - topics: List of Topic objects
        '''
        self.topics = topics
        self.buffer_capacity = buffer_capacity
        self.cvbridge = CvBridge()
        self.subscribers = [
            rospy.Subscriber(self.topics[i].topic_name, 
                             self.topics[i].msg_type, 
                             self.topics[i].io_function, 
                             queue_size=1) for i in range(len(self.topics)
                            )]
        
        self.obs_space, self.act_space = _obs_act_space_from_topics(self.topics)
        
        self.replay_buffer = ReplayBuffer(self.obs_space, self.act_space, buffer_capacity)

    def 
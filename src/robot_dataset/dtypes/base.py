"""
Generic template for what we need from each type for this class.
"""

import abc

class Dtype:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def N(self):
        """
        Get the shape of the output
        """
        pass

    @abc.abstractmethod
    def obs_space(self):
        """
        Get the gym observation space that will be used for the replay buffer.
        """
        pass

    @abc.abstractmethod
    def action_space(self):
        """
        Get the gym action space that will be used for the replay buffer.
        """
        pass

    @abc.abstractmethod
    def rosmsg_type(self):
        """
        Get the type of the rosmsg we should be reading
        Note that certain dtypes may have multiple rosmsgs. I'm not sire how to handle that cleanly yet.
        """
        pass

    @abc.abstractmethod
    def ros_to_numpy(self, msg):
        """
        Convert an instance of the rsomsg to a numpy array.
        """
        pass

    @abc.abstractmethod
    def save_file_one_msg(self, data, filename):
        """
        Save the data to hard drive.
        This function should be implemented where the data is stored frame by frame like image or point cloud
        """
        pass

    @abc.abstractmethod
    def save_file(self, data, filename):
        """
        Save the data to hard drive.
        This function should be implemented where the data of the whole trajectory is stored in one file, like imu, odometry, etc.
        """
        pass

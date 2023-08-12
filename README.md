Replay buffer for continual learning using similar infrastructure to [JaxRL and JaxRL 2](https://github.com/ikostrikov/jaxrl2) (Kostrikov et al.), as well as [rosbag_to_dataset](https://github.com/striest/rosbag_to_dataset) by Sam Triest

To run online HDIF, do the following:

1. Make sure all packages are installed and set up(including WandB for visualization)
2. Set up a spec for your desired application similar to hdif_lester.yaml, which is in the specs folder.
3. Run all the desired nodes that will publish ROS topics with the relevant data. If done from a bag, make sure to set simulation time to true with the following parameter (after running roscore): ```rosparam set use_sim_time true```
4. Change the desired hdif configuration in ```scripts/online_hdif.py```, including map_config, saved_model, saved_freqs, and saved_dir.
5. Run alongside data-generating nodes (such as tartanvo), and check the losses on WandB.
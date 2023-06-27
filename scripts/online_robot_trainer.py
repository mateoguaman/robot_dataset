#!/usr/bin/python3
import rospy
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pickle

from robot_dataset.data.replay_buffer import ReplayBuffer
from robot_dataset.online_converter.robot_listener import RobotListener

USE_WANDB = True

def quaternionToEuler(q):
    '''Assumes q = [x, y, z, w], returns [roll, pitch, yaw]'''
    qx, qy, qz, qw = q[:,0], q[:,1], q[:,2], q[:,3]

    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.sqrt(1 + 2*(qw*qy - qx*qz))
    cosp = np.sqrt(1 - 2*(qw*qy - qx*qz))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi/2

    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


class MLP(nn.Module):
    def __init__(self, state_dim=3, action_dim=2, latent_dim=32):
        super().__init__()
        self.state_model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
        )

        self.action_model = nn.Sequential(
            nn.Linear(in_features=action_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim)
        )

        self.cat_model = nn.Sequential(
            nn.Linear(in_features=2*latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=state_dim)
        )

    def forward(self, state, action):
        state_out = self.state_model(state)
        action_out = self.action_model(action)
        combined = torch.concatenate([state_out, action_out], dim=-1)
        out = self.cat_model(combined)
        return out
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

    def forward(self, x):
        out = self.model(x)
        return out
    
def transform(batch):
    obs = batch['observations']
    next_obs = batch['next_observations']
    actions = torch.from_numpy(batch['actions'])

    odom = obs['odom']
    next_odom = next_obs['odom']

    # imgs = obs['image_left_color']
    # next_imgs = obs['image_left_color']

    # resnet_transform = ResNet18_Weights.DEFAULT.transforms()
    # trans_imgs = resnet_transform(imgs)
    # trans_next_imgs = resnet_transform(next_imgs)

    px, py, yaw = odom[:,0], odom[:,1], quaternionToEuler(odom[:,3:7])[:,-1]
    next_px, next_py, next_yaw = next_odom[:,0], next_odom[:,1], quaternionToEuler(next_odom[:,3:7])[:, -1]

    # import pdb;pdb.set_trace()

    pose = np.stack([px, py, yaw], axis=-1)
    next_pose = np.stack([next_px, next_py, next_yaw], axis=-1)
    pose_diff = next_pose - pose

    states = torch.from_numpy(pose)
    ground_truth = torch.from_numpy(pose_diff)

    return states, actions, ground_truth
    
class Agent:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def update(self, model, batch, data_transform, optimizer, device="cuda"):
        model.train()
        optimizer.zero_grad()
        states, actions, ground_truth = data_transform(batch)
        states = states.to(device)
        actions = actions.to(device)
        ground_truth = ground_truth.to(device)
        next_state_diffs = model(states, actions)
        loss = self.criterion(ground_truth, next_state_diffs)
        loss.backward()
        optimizer.step()

        training_metrics = {
            'loss': loss.item()
        }
        return training_metrics

def main():
    ## Define data source (RobotListener or Simulation Environment)
    # Load spec and get parser
    config_spec = "/home/mateo/robot_dataset/specs/sample_tartandrive.yaml"
    rospy.init_node('online_trainer')
    rate = rospy.Rate(10)
    robot_listener = RobotListener(config_spec=config_spec)
    print("Robot Listener set up")

    ## Define agent similar to BC (take in model, take in batch, return MSE loss)
    model = MLP(state_dim=3, action_dim=2, latent_dim=32)
    print("Model defined")
    data_transform = transform
    agent = Agent()
    print("Agent defined")

    model.to("cuda")
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    ## Instantiate replay buffer 
    buffer_capacity = 200
    replay_buffer = ReplayBuffer(robot_listener.obs_space, robot_listener.action_space, buffer_capacity)
    print("Replay Buffer defined")

    ## Training loop. Is there a way to make it asynchronous? Threading? One for gathering data into replay buffer, one for training
    lifetime = 1000000
    count = 1
    replay_ratio = 10  ## For now, it is the number of batches obtained from the dataloader and trained at every training step within data collection
    batch_size = 64
    training_freq = 10
    save_freq = 10
    save_dir = "/media/mateo/MateoSSD/online_training"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config = {
        'lr': lr,
        'buffer_capacity': buffer_capacity, 
        'replay_ratio': replay_ratio,
        'batch_size': batch_size,
        'training_freq': training_freq,
        'save_freq': save_freq
    }
    wandb.init(project="online_learning", reinit=True, config=config)

    print('waiting 2s for topics...')
    for i in range(10):
        rate.sleep()

    while (not rospy.is_shutdown()) and (count < lifetime):
        print(f"Iteration {count}/{lifetime}")
        robot_data = robot_listener.get_data()
        replay_buffer.insert(robot_data)

        if count % training_freq == 0:  ## Train network
            ## Extract batches in for loop
            for i in range(replay_ratio):
                batch = replay_buffer.sample(batch_size=batch_size)
                train_metrics = agent.update(model, batch, data_transform, optimizer)
                loss = train_metrics["loss"]
                print(f"Training loss: {loss}")
                wandb.log(data=train_metrics, step=count)

        ## Save model and buffer every k iterations
        if (count % save_freq == 0):
            buffer_folder = os.path.join(save_dir, "datasets")
            if not os.path.exists(buffer_folder):
                os.makedirs(buffer_folder)
            dataset_file = os.path.join(buffer_folder, f"buffer_{count}.pickle")
            model_folder = os.path.join(save_dir, "models")
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_file = os.path.join(model_folder, f"model_{count}.pt")
            with open(dataset_file, "wb") as f:
                pickle.dump(replay_buffer, f)

            torch.save(model.state_dict(), model_file)
        count += 1
        rate.sleep()

    ## Update logging

if __name__ == "__main__":
    main()
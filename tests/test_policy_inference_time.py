import torch
import os
import numpy as np
import cv2
import time
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../../hannes-imitation/')
sys.path.append('../../hannes-imitation/hannes_imitation/external/diffusion_policy/') # NOTE otherwise importing SequenceSampler fails

# diffusion_policy imports
from hannes_imitation.external.diffusion_policy.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from hannes_imitation.scripts.validate_policy import create_observation_dictionaries


# hannes_imitation imports
from hannes_imitation.dataset.hannes_dataset import HannesImageDataset
from hannes_imitation.dataset.hannes_dataset_hand_wrist_FE import HannesImageDatasetWrist

def get_policy_inference_times(policy, episode):
    obs_dictionaries, actions_gt = create_observation_dictionaries(episode, observation_horizon=policy.n_obs_steps)

    inference_times = []
    # evaluate policy on episode
    with torch.no_grad():
        for obs_dict in obs_dictionaries:
            # predict action trajectory
            tic = time.time()
            action_predictions = policy.predict_action(obs_dict) # {'action', 'action_pred}
            toc = time.time()
        
            inference_times.append(toc - tic)
    
    return inference_times


# Load train/validation dataset
zarr_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/training/merged_1_4_7.zarr' # NOTE: IROS 2025
keys = ['image_in_hand', 'ref_move_hand', 'ref_move_wrist_FE', 'ref_move_wrist_PS', 'mes_hand', 'mes_wrist_FE']
val_ratio = 0.2
seed = 72
max_train_episodes = None
horizon = 8 # default 16 # prediction horizon
observation_horizon = 2 # default 2
action_horizon = 4 # default 8
pad_before = observation_horizon - 1
pad_after = action_horizon - 1

# training and validation dataset
train_dataset = HannesImageDatasetWrist(zarr_path, keys, obs_horizon=observation_horizon, horizon=horizon, pad_before=pad_before, pad_after=pad_after, 
                                        seed=seed, val_ratio=val_ratio, max_train_episodes=None)
validation_dataset = train_dataset.get_validation_dataset()

# get global episode indeces for the training set and validation set
tr_episode_indeces = [i for i, mask in enumerate(train_dataset.train_mask) if mask]
vl_episode_indeces = [i for i, mask in enumerate(validation_dataset.train_mask) if mask]

# load model
policy_path = '/home/calessi-iit.local/Projects/hannes-imitation/trainings/policy_1_4_7_2025_2_19-21_10_9.pth' # iros2025
checkpoint = torch.load(policy_path)
policy = checkpoint['policy']

# device transfer
device = torch.device('cuda')
_ = policy.to(device).eval()


# dummy inference to setup pytorch
ep_idx = np.random.choice(vl_episode_indeces)
episode = train_dataset.replay_buffer.get_episode(ep_idx)
obs_dictionaries, actions_gt = create_observation_dictionaries(episode, observation_horizon=policy.n_obs_steps)

with torch.no_grad():
    for i in range(10):
        # predict action trajectory
        tic = time.time()
        action_predictions = policy.predict_action(obs_dictionaries[0]) # {'action', 'action_pred}
        toc = time.time()
        print("Trial %d, inference time: %.3f s" % (i, toc - tic))

# evaluate inference time
ep_idx = np.random.choice(vl_episode_indeces)
episode = train_dataset.replay_buffer.get_episode(ep_idx)
inference_times = get_policy_inference_times(policy, episode)

print("====================")
print("Diffusion iterations:", policy.num_inference_steps) # policy.noise_scheduler.num_train_timesteps)
print("N. prediction samples: %d" % len(inference_times))
print("Policy inference time: %.3f +- %.5f s" % (np.mean(inference_times), np.std(inference_times)))
print("Policy inference frequency: %.1f Hz" % (1/np.mean(inference_times)))

"""
Old measures with different network architecture
Diffusion iterations: 50
Avg prediction time: 0.2 +- 0.143 s
Avg prediction frequency: 4.3 Hz

Diffusion iterations: 100
Avg prediction time: 0.5 +- 0.026 s
Avg prediction frequency: 1.9 Hz
"""
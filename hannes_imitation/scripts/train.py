import os
import zarr
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import time

import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation')
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation/hannes_imitation/external/diffusion_policy') # NOTE otherwise importing SequenceSampler fails

# diffusion_policy imports
from hannes_imitation.external.diffusion_policy.diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from hannes_imitation.external.diffusion_policy.diffusion_policy.model.vision.model_getter import get_resnet
from hannes_imitation.external.diffusion_policy.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from hannes_imitation.external.diffusion_policy.diffusion_policy.model.common.lr_scheduler import get_scheduler

# hannes_imitation imports
#from hannes_imitation.dataset.hannes_dataset import HannesImageDataset
from hannes_imitation.dataset.hannes_dataset_hand_wrist_FE import HannesImageDatasetWrist
from hannes_imitation.trainer.trainer_diffusion_policy import TrainerDiffusionPolicy
from hannes_imitation.common import plot_utils

# diffusers import
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

#zarr_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/training/merged-grasp-ycb-table.zarr'
#zarr_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/training/merged_collections_1_4.zarr'
#zarr_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/training/merged_collections_1_3_4_6.zarr'
#zarr_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/training/merged_collections_1-8.zarr'
zarr_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/training/merged_1_4_7.zarr' # NOTE: IROS 2025
keys = ['image_in_hand', 'ref_move_hand', 'ref_move_wrist_FE', 'ref_move_wrist_PS', 'mes_hand', 'mes_wrist_FE']
val_ratio = 0.2 #0.25
seed = 72
max_train_episodes = None
horizon = 8 # default 16 # prediction horizon
observation_horizon = 2 # default 2
action_horizon = 4 # default 8
pad_before = observation_horizon - 1
pad_after = action_horizon - 1

# training and validation dataset
#train_dataset = HannesImageDataset(zarr_path, keys, horizon=horizon, pad_before=pad_before, pad_after=pad_after, seed=seed, val_ratio=val_ratio, max_train_episodes=None)
train_dataset = HannesImageDatasetWrist(zarr_path, keys, obs_horizon=observation_horizon, horizon=horizon, pad_before=pad_before, pad_after=pad_after, seed=seed, val_ratio=val_ratio, max_train_episodes=None)
validation_dataset = train_dataset.get_validation_dataset()

# get normalizer
normalizer = train_dataset.get_normalizer()

# create dataloaders for training and validation
batch_size = 128 # default 64 #128 per prima iros2025 
num_workers = 4
shuffle = True

# pin_memory = True accelerates cpu-gpu transfer
# persistent_workers = True does not kill worker process after each epoch
tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
vl_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)

# visualize data in batch
print("======== data loader ===============")
batch = next(iter(tr_dataloader))
print("N. batches", len(tr_dataloader))
print('Observations:')
for key, obs_value in batch['obs'].items():
    print(key, obs_value.shape, obs_value.dtype)
print("Action:")
print(batch['action'].shape, batch['action'].dtype)
print("=======================")

# Create shape_meta
item = train_dataset.__getitem__(0)
_, C, H, W = item['obs']['image_in_hand'].shape
_, action_dim = item['action'].shape

# Make sure that `shape_meta` correspond to input and output shapes for your task.
shape_meta = dict(obs=dict(), action=dict())
shape_meta['obs']['image_in_hand'] = dict(shape=(C, H, W), type='rgb')
shape_meta['obs']['mes_hand'] = dict(shape=[1], type='low_dim')
shape_meta['obs']['mes_wrist_FE'] = dict(shape=[1], type='low_dim')
shape_meta['action'] = dict(shape=[action_dim])

# Create observation encoder
rgb_model = get_resnet('resnet18') # dict()

# The MultiImageObsEncoder encodes image and low dimensional observations into a single observation.
# The constructor requires 2 positional arguments (shape_meta, rgb_model).
# rgb_model can be directly an nn.Module or a Dict[str,nn.Module]

# Optionally, you can specify if the image is resized (`resize_shape`) and/or cropped (`crop_shape`, `random_crop`) and/or
# normalized according to imagenet values (`imagenet_norm`)
# These transformations are performed in the forward() method.
# We only use imagenet_norm.
observation_encoder = MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=rgb_model,
                                       resize_shape=None,
                                       crop_shape=None,
                                       random_crop=False,
                                       use_group_norm=True,
                                       share_rgb_model=True,
                                       imagenet_norm=True)

# freeze observation_encoder
_ = observation_encoder.eval()

# Create noise scheduler
# for this demo, we use DDPMScheduler with 100 diffusion iterations
# NOTE: the choice of beta schedule has big impact on performance. We found squared cosine works the best
num_diffusion_iters = 10 # default 100 
noise_scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                                beta_schedule='squaredcos_cap_v2',
                                clip_sample=True, # clip output to [-1,1] to improve stability
                                prediction_type='epsilon') # the network predicts noise (instead of denoised action)

device = torch.device('cuda')

# DiffusionUnetImagePolicy requires 6 positional arguments
# shape_meta is a dictionary that contains the shapes of observations and actions for the task
# noise_scheduler is an instance of DDPMScheduler noise scheduler
# obs_encoder is an instance of MultiImageObsEncoder which encodes images and low dimensional observations as conditioning
# horizon is the prediction horizon (action prediction horizon)s
# n_action_steps is the action execution horizon (how many actions are actually executed from the prediction)
# n_obs_steps is the observation horizon (how many recent observations to include as condition)

# NOTE: there are other parameters that we do not change, except for the UNet model size
policy = DiffusionUnetImagePolicy(shape_meta=shape_meta, 
                                  noise_scheduler=noise_scheduler, 
                                  obs_encoder=observation_encoder,
                                  horizon=horizon,
                                  n_action_steps=action_horizon,
                                  n_obs_steps=observation_horizon,
                                  diffusion_step_embed_dim=32,#64, #128,#256 default,
                                  down_dims=[32, 64], # [16, 32, 64])#(256,512,1024)) default
                                  kernel_size=3, # default 5
                                  n_groups=8, # default 8
                                  cond_predict_scale=True) # default True 

# Create optimizer and learning rate scheduler
# Standard ADAM optimizer (NOTE that EMA parametesr are not optimized)
optimizer = torch.optim.AdamW(params=policy.parameters(), lr=1e-4, weight_decay=2e-4) # default weigth_decay=1e-6

num_epochs = 100

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(tr_dataloader) * num_epochs)

# Train policy
policy_trainer = TrainerDiffusionPolicy(policy=policy, 
                                        optimizer=optimizer, 
                                        normalizer=normalizer, 
                                        tr_dataloader=tr_dataloader, 
                                        vl_dataloader=vl_dataloader, 
                                        learning_rate_scheduler=lr_scheduler)
history = policy_trainer.run(num_epochs=num_epochs, device=device)

local_time = time.localtime()
timestamp = '_%d_%d_%d-%d_%d_%d' % (local_time.tm_year, local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

# save policy and training results
policy_dir = '/home/calessi-iit.local/Projects/hannes-imitation/trainings/'
#policy_name = 'preliminary_policy.pth'
#policy_name = 'preliminary_policy_wrist_FE-tmp.pth'
policy_name = 'policy_1_4_7' + timestamp + '.pth'
policy_path = os.path.join(policy_dir, policy_name)

training_dict = {'policy': policy.to('cpu'),
                 'policy_state_dict': policy.state_dict(),
                 'optimizer': optimizer,
                 'noise_scheduler': noise_scheduler,
                 'history': history
                }

torch.save(training_dict, policy_path)

print("Training saved in %s" % str(policy_path))
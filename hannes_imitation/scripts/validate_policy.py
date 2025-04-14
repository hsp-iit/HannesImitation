"""
Evaluates a policy on the validation set.
Saves the action errors for each episode.
"""

import torch
import os
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../../../hannes-imitation/')
sys.path.append('../../../hannes-imitation/hannes_imitation/external/diffusion_policy/') # NOTE otherwise importing SequenceSampler fails

# diffusion_policy imports
from hannes_imitation.external.diffusion_policy.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply

# hannes_imitation imports
from hannes_imitation.dataset.hannes_dataset import HannesImageDataset
from hannes_imitation.dataset.hannes_dataset_hand_wrist_FE import HannesImageDatasetWrist

from hannes_imitation.common import plot_utils
from hannes_imitation.common.data_utils import resize_image


def create_observation_dictionaries(episode, observation_horizon):
    # TODO: maybe also for actions
    frames = episode['image_in_hand']
    mes_hand = episode['mes_hand']
    mes_wrist_FE = episode['mes_wrist_FE']
    actions_gt = np.concatenate((episode['ref_move_hand'], episode['ref_move_wrist_FE'], episode['ref_move_wrist_PS']), axis=1)

    # create observation dictionaries
    # NOTE: unfold creates the new dimensions as last; then we move it in second position
    frames = torch.from_numpy(frames).unfold(dimension=0, size=observation_horizon, step=1).moveaxis(source=-1, destination=1) # (Time, To, H, W, C)
    mes_hand = torch.from_numpy(mes_hand).unfold(dimension=0, size=observation_horizon, step=1).moveaxis(source=-1, destination=1) # (Time, To, 1)
    mes_wrist_FE = torch.from_numpy(mes_wrist_FE).unfold(dimension=0, size=observation_horizon, step=1).moveaxis(source=-1, destination=1) # (Time, To, 1)

    # only for frames move C before H, W and scale to 0-1. (other preprocessing done inside the policy)
    frames = frames.moveaxis(source=-1, destination=-3) # (Time, To, C, H, W)
    frames = frames.float() / 255.0
    mes_hand = mes_hand.float()
    mes_wrist_FE = mes_wrist_FE.float()

    # create list of observation dictionaries
    obs_dictionaries = []
    for i in range(len(mes_hand)):
        obs_dict = {
            'image_in_hand': frames[i],
            'mes_hand': mes_hand[i],
            'mes_wrist_FE': mes_wrist_FE[i]}
        obs_dictionaries.append(obs_dict)

    # prepend dummy batch dimension
    for obs_dict in obs_dictionaries:
        for key, item in obs_dict.items():
            obs_dict[key] = item.unsqueeze(0)

    return obs_dictionaries, actions_gt

def evaluate_policy_on_episode(policy, episode, observation_horizon):
    obs_dictionaries, actions_gt = create_observation_dictionaries(episode, observation_horizon)
    
    # evaluate policy on episode
    action_sequences = {'predicted': [], 'executed': []}

    with torch.no_grad():
        for obs_dict in obs_dictionaries:
            # predict action trajectory
            action_predictions = policy.predict_action(obs_dict) # {'action', 'action_pred}
            
            predicted_action_sequence = action_predictions['action_pred'].cpu().detach().numpy() # (B, T, Da)
            executed_action_sequence = action_predictions['action'].cpu().detach().numpy() # (B, Ta, Da)
            
            action_sequences['predicted'].append(predicted_action_sequence.squeeze())
            action_sequences['executed'].append(executed_action_sequence.squeeze())

    return action_sequences, actions_gt

def compute_episode_errors_per_horizon(gt_actions, pred_actions, observation_horizon, h):
    # for horizon h=0 you don't need to cut anything, else you need to cut h samples from the end of the predictions
    last_index = len(pred_actions) if h == 0  else - h
    A_hat = pred_actions[:last_index, h]

    # for the ground truth, you skip the first observation_horizon-1 samples (because here we don't do padding), and select the right horizon.
    A = gt_actions[observation_horizon-1+h:]

    errors = A - A_hat # (T,3)

    return errors

if __name__ == '__main__':
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

    tr_labels = [train_dataset.labels[i] for i in tr_episode_indeces]
    vl_labels = [train_dataset.labels[i] for i in vl_episode_indeces]

    # load model
    policy_path = '/home/calessi-iit.local/Projects/hannes-imitation/trainings/policy_1_4_7_2025_2_19-21_10_9.pth' # iros2025
    checkpoint = torch.load(policy_path)
    policy = checkpoint['policy']

    # device transfer
    device = torch.device('cuda')
    _ = policy.to(device).eval()


    results_dicts = []

    for i, ep_idx in enumerate(tqdm(vl_episode_indeces)):
        episode = train_dataset.replay_buffer.get_episode(ep_idx)
        action_sequences, actions_gt = evaluate_policy_on_episode(policy, episode, observation_horizon=policy.n_obs_steps) # {'executed', 'predicted'}

        executed_actions = np.array(action_sequences['executed'])
        errors = compute_episode_errors_per_horizon(actions_gt, pred_actions=executed_actions, observation_horizon=observation_horizon, h=0)

        # append results
        results_dicts.append(vl_labels[i])
        results_dicts[-1]['errors'] = errors


    np.savez('/home/calessi-iit.local/Projects/hannes-imitation/data/validation/validation_set_results.npz', **{f'dict_{i}': d for i, d in enumerate(results_dicts)})

    print("Validation results saved at %s" % '/home/calessi-iit.local/Projects/hannes-imitation/data/validation/validation_set_results.npz')
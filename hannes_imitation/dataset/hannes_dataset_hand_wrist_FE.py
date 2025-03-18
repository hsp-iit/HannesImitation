from hannes_imitation.external.diffusion_policy.diffusion_policy.dataset.base_dataset import BaseImageDataset
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from hannes_imitation.external.diffusion_policy.diffusion_policy.model.common.normalizer import LinearNormalizer
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.normalize_util import get_image_range_normalizer
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply

from typing import Dict
import copy
import torch
import numpy as np
import zarr

class HannesImageDatasetWrist(BaseImageDataset):
    def __init__(self, zarr_path, keys, obs_horizon=2, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__()
        self.obs_horizon = obs_horizon

        # Create the replay buffer
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)#['img', 'state', 'action'])

        # Create a train and validation mask for the episodes in the replay buffer
        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed) # True in the val_mask means the episode is used for validation
        train_mask = ~ val_mask # train mask is the opposite of the val_mask
        # train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed) # NOTE: we ignore it. Also, use max_train_episodes=None

        # Create the sequence sampler
        sampler = SequenceSampler(replay_buffer=replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)

        # Assign instance variables
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.labels = zarr.open(zarr_path, 'r').attrs['label'][:]

    # --------------------------------------------------------------
    # ------- Methods inherited by BaseImageDataset ----------------
    # --------------------------------------------------------------

    def get_validation_dataset(self):
        """
        Creates a validation dataset from the entire dataset.
        It creates a new sequence sampled with a validation mask (i.e., the negated train_mask).
        
        Returns a HannesImageDataset to be used in validation.
        """
        # Create a copy of itself inverting the training mask
        validation_set = copy.copy(self)
        validation_mask = ~ self.train_mask

        # Redefine the sequence sampler using the validation mask as the episode mask
        validation_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=validation_mask)
        
        validation_set.train_mask = validation_mask

        return validation_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        """
        Creates and returns a normalizer.
        The normalizer is fit to the data.
        """
        ref_move_hand = self.replay_buffer['ref_move_hand']
        ref_move_wrist_FE = self.replay_buffer['ref_move_wrist_FE']
        ref_move_wrist_PS = self.replay_buffer['ref_move_wrist_PS']

        action = np.concatenate((ref_move_hand, ref_move_wrist_FE, ref_move_wrist_PS), axis=1)

        data = {
            'mes_hand': self.replay_buffer['mes_hand'],
            'mes_wrist_FE': self.replay_buffer['mes_wrist_FE'],
            'action': action,
        }

        # Create Normalizer
        normalizer = LinearNormalizer()

        # fit non-image data in range -1, 1. 
        # TODO: understand
        # (last_n_dims: int = 1 dtype: dtype = torch.float32, mode: str = 'limits', output_max: float = 1, output_min: float = -1, range_eps: float = 0.0001, fit_offset: bool = True)
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # image data are fitted differently TODO: understand
        normalizer['image_in_hand'] = get_image_range_normalizer()

        return normalizer
    
    # --------------------------------------------------------------
    # ------- Methods inherited by torch.utils.Dataset -------------
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        NOTE: images are 0-1, action are not scaled
        """
        # Get one sample from the sequence sampler
        # This will be a dict of numpy arrays
        # {'image_in_hand': shape(horizon, H, W, C)
        #  'ref_move_hand: shape(horizon, Da)}
        sample = self.sampler.sample_sequence(idx)

        torch_sample = self._convert_sample_to_torch_data(sample)

        # discard unused observations
        #torch_sample['obs']['image_in_hand'] = torch_sample['obs']['image_in_hand'][:, self.sampler.ob]
        #self.replay_buffer
        
        return torch_sample
    
    def _convert_sample_to_torch_data(self, sample):
        """
        Converts sample to torch data following the interface 
        {'obs': Dict[str, torch.Tensor],
         'action' torch.Tensor}

        NOTE: data should be float32
        """
        # bring channel axis in the first position (horizon, H, W, C) -> (horizon, C, H, W)
        # also scale images from 0-255 to 0-1
        image = sample['image_in_hand']
        image = np.moveaxis(image, source=-1, destination=1)
        image = image.astype(np.float32) / 255.0

        mes_hand = sample['mes_hand'].astype(np.float32)
        mes_wrist_FE = sample['mes_wrist_FE'].astype(np.float32)
        
        # unsqueeze dimension of ref_move_hand action (horizon,) -> (horizon, 1)
        ref_move_hand = sample['ref_move_hand'].astype(np.float32)
        ref_move_wrist_FE = sample['ref_move_wrist_FE'].astype(np.float32)
        ref_move_wrist_PS = sample['ref_move_wrist_PS'].astype(np.float32)

        action = np.concatenate((ref_move_hand, ref_move_wrist_FE, ref_move_wrist_PS), axis=1)


        data = {
            'obs': {
                'image_in_hand': image[:self.obs_horizon], # T, 3, H, W NOTE: images were (96,96) in PushT
                'mes_hand': mes_hand[:self.obs_horizon],
                'mes_wrist_FE': mes_wrist_FE[:self.obs_horizon],                
            },
            'action': action # T, 2
        }

        # transform each numpy array in the data dictionary into a torch.Tensor
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data
import time
import numpy as np
import cv2
import torch

# hannes_imitation imports
from hannes_imitation.common.data_utils import resize_image

# diffusion_policy imports
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply


class HannesDiffusionPolicyController:

    def __init__(self, hannes, policy, control_frequency, observation_horizon, action_horizon, frames_list, hannes_states_list, lock):
        self.hannes = hannes
        self.policy = policy
        self.control_frequency = control_frequency
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.frames_list = frames_list
        self.hannes_states_list = hannes_states_list
        self.lock = lock

    def get_observation(self):
        # TODO: this function could be in env wrapper
        
        ## GET CAMERA OBSERVATIONS
        # spin while there are not enough frames
        obs_frames_raw = [] # start with 0 frames

        while len(obs_frames_raw) != self.observation_horizon:
            # NOTE essendo self.frames_list un multiprocessing.manager.list non dovrebbe servire la lock in quanto concorrenza già gestita 
            with self.lock:
                obs_frames_raw = np.array(self.frames_list[-self.observation_horizon:]) # read the latest obs_horizon frames
        
        # preprocess frames
        obs_frames = [resize_image(frame, scaling_factor=0.2) for frame in obs_frames_raw] # rescale as images used for training
        obs_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in obs_frames] # convert brg of openCV to rgb
        obs_frames = np.moveaxis(obs_frames, source=-1, destination=1) # move Channel dimension before W,H
        obs_frames = obs_frames.astype(np.float32) / 255.0 # scale to 0-1
        obs_frames = np.expand_dims(obs_frames, axis=0) # include batch size in first dimension
            
        ## GET ENCODERS OBSERVATIONS
        latest_hannes_states = [] # hold latest observation_horizon hand states
        while len(latest_hannes_states) != self.observation_horizon:
            latest_hannes_states = list(self.hannes_states_list)[-self.observation_horizon:]

        mes_hand = np.array([hannes_state['joints']['hand']['position'] for hannes_state in latest_hannes_states], dtype=np.float32)
        mes_wrist_FE = np.array([hannes_state['joints']['wrist_FE']['position'] for hannes_state in latest_hannes_states], dtype=np.float32)

        mes_hand = mes_hand.reshape((1, self.observation_horizon, -1)) # include batch size first and measurement dimension last
        mes_wrist_FE = mes_wrist_FE.reshape((1, self.observation_horizon, -1)) # include batch size first and measurement dimension last

        # create observation dictionary
        obs_dict = {'image_in_hand': obs_frames,
                    'mes_hand': mes_hand,
                    'mes_wrist_FE': mes_wrist_FE}
        obs_dict = dict_apply(obs_dict, torch.from_numpy)

        return obs_dict
    
    def predict(self, obs_dict):
        action_predictions = self.policy.predict_action(obs_dict) # {'action', 'action_pred}

        action_trajectory = action_predictions['action'].cpu().detach().numpy() # (B, Ta, Da)
        action_trajectory = np.array(action_trajectory).reshape(self.action_horizon, -1) # remove batch dimension (Ta, Da)

        return action_trajectory
    

    def step(self, action_trajectory):
        # actuate hand with action trajectory
        for action in action_trajectory:
            self.hannes.move_hand(int(action[0]))
            self.hannes.move_wristFE(int(action[1]))
            
            time.sleep(1.0 / self.control_frequency)
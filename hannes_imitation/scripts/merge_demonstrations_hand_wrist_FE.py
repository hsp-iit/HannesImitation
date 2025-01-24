import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation')

from hannes_imitation.common.data_utils import filter_demonstrations, index_alignment, resize_image, store_merged_data_to_zarr
from hannes_imitation.common.config import load_configuration
import os
import zarr
import numpy as np
from tqdm import tqdm

# load configuration file
config_path = '/home/calessi-iit.local/Projects/hannes-imitation/hannes_imitation/config/data/config_preliminary_hand_wrist_FE.yaml'
config = load_configuration(config_path)

# extract configuration parameters
demonstration_dir = config['directory']['source']
merged_dir = config['directory']['destination']
merged_name = config['directory']['merged_name']
merged_data = config['merged_data']
scaling_factor = config['scaling_factor']

print("=================")
print('Source:', demonstration_dir)
print('Dest:', merged_dir)
print('Merged file:', merged_name)
print(merged_data)
print('Scaling factor:', scaling_factor)

# filter demonstrations
demonstration_names = filter_demonstrations(demonstration_dir)

print("=================")
print("Demonstrations filtered.")
print("Demonstrations retained:", len(demonstration_names['retained']))
print("Demonstrations discarded:", len(demonstration_names['discarded']))

for demonstration_name in tqdm(demonstration_names['retained']):
    # open zarr store of episode
    store = zarr.open(demonstration_dir + demonstration_name)

    # extract action and observations needed
    ref_move_hand = store['references']['hand'][:]
    ref_move_wrist_FE = store['references']['wrist_FE'][:]
    mes_hand = store['joints']['hand']['position'][:]
    mes_wrist_FE = store['joints']['wrist_FE']['position'][:]
    camera = store['in-hand_camera_0']['frames'][:]

    # get desired indeces for temporal alignment for hand time stamps and video time stamps
    hand_times = store['time'][:]
    camera_times = store['in-hand_camera_0']['time'][:]
    time_stamps_collection = {'hand': hand_times, 
                              'camera': camera_times}
    indeces_to_select = index_alignment(x_collection=time_stamps_collection)

    # extract desired indeces for temporal alignment
    ref_move_hand = ref_move_hand[indeces_to_select['hand']]
    ref_move_wrist_FE = ref_move_wrist_FE[indeces_to_select['hand']]
    mes_hand = mes_hand[indeces_to_select['hand']]
    mes_wrist_FE = mes_wrist_FE[indeces_to_select['hand']]
    camera = camera[indeces_to_select['camera']]
    episode_lenth = len(indeces_to_select['hand'])
    
    # downsize all frames
    camera = [resize_image(frame, scaling_factor=scaling_factor) for frame in camera]

    # add to merged dataset
    merged_data['data']['ref_move_hand'].extend(ref_move_hand.reshape(-1, 1)) # reshape ref_move_hand (T,) -> (T,1)
    merged_data['data']['ref_move_wrist_FE'].extend(ref_move_wrist_FE.reshape(-1, 1))
    merged_data['data']['mes_hand'].extend(mes_hand.reshape(-1, 1))
    merged_data['data']['mes_wrist_FE'].extend(mes_wrist_FE.reshape(-1, 1))
    merged_data['data']['image_in_hand'].extend(camera)
    merged_data['meta']['episode_ends'].append(episode_lenth)

# compute episode ends form list of episode lengths
# NOTE I had put -1 to make the episode ends match with array indeces, but when creating their ReplayBuffer you get an assertion error.
merged_data['meta']['episode_ends'] = np.cumsum(merged_data['meta']['episode_ends']) # - 1

# save merged data in new file
merged_path = os.path.join(merged_dir, merged_name)
store_merged_data_to_zarr(merged_data, path=merged_path)

print("=================")
print("Saved merged store at %s" % merged_path)


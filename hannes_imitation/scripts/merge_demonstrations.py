import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation')

from hannes_imitation.common.data_utils import filter_demonstrations, index_alignment, resize_image, store_merged_data_to_zarr
from hannes_imitation.common.config import load_configuration
import os
import zarr
import numpy as np
from tqdm import tqdm
import cv2

# load configuration file
#config_path = '/home/calessi-iit.local/Projects/hannes-imitation/hannes_imitation/config/data/config_hand_wrist.yaml'
config_path = '/home/calessi-iit.local/Projects/hannes-imitation/hannes_imitation/config/data/config_hand_wrist_IROS2025.yaml' 
config = load_configuration(config_path)

# extract configuration parameters
source_dir = config['directory']['source']
merged_dir = config['directory']['destination']
merged_name = config['directory']['merged_name']
merged_data = config['merged_data']
scaling_factor = config['scaling_factor']

merged_path = os.path.join(merged_dir, merged_name)

print("=================")
print('Source:')
for dir in source_dir:
     print(dir)
print('Dest:', merged_dir)
print('Merged file:', merged_name)
print('Merged path:', merged_path)
print(merged_data)
print('Scaling factor:', scaling_factor)

# filter demonstrations
demonstration_names = {}
for dir in source_dir:
    demonstration_names[dir] = filter_demonstrations(dir)

print("======== filtered demonstrations =========")
for dir in source_dir:
    print("Source:", dir)
    print("Demonstrations retained:", len(demonstration_names[dir]['retained']))
    print("Demonstrations discarded:", len(demonstration_names[dir]['discarded']))

# merge demonstration paths from all source directories
demonstrations_paths = []
for source_directory, names in demonstration_names.items():
     for demo_name in names['retained']:
          demo_path = os.path.join(source_directory, demo_name)
          demonstrations_paths.append(demo_path)

def __create_zarr_groups(group, dictionary):
        """
        Recursively add dictionary data to Zarr group.
        We use it to initialize the zarr groups with the hannes state.
        Call it as add_to_zarr_group(group=self.store, dictionary=hannes_state)
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # there is a nested dictionary, create a group/subgroup
                print(key, value)
                subgroup = group.create_group(key)
                __create_zarr_groups(subgroup, value)
            else:
                # else, just add values/arrays to the group
                group.array(key, zarr.array(value))


merged_store = zarr.open(merged_path, 'w')
merged_store.attrs['label'] = []
__create_zarr_groups(group=merged_store, dictionary=merged_data)

print(merged_store.tree())

for i, demonstration_path in tqdm(enumerate(demonstrations_paths)):
    # open zarr store of episode
    store = zarr.open(demonstration_path)

    # extract action and observations needed
    ref_move_hand = store['references']['hand'][:]
    ref_move_wrist_FE = store['references']['wrist_FE'][:]
    ref_move_wrist_PS = store['references']['wrist_PS'][:]
    mes_hand = store['joints']['hand']['position'][:]
    mes_wrist_FE = store['joints']['wrist_FE']['position'][:]
    camera = store['in-hand_camera_0']['rgb'][:]

    # get desired indeces for temporal alignment for hand time stamps and video time stamps
    hand_times = store['time'][:]
    camera_times = store['in-hand_camera_0']['time'][:]
    time_stamps_collection = {'hand': hand_times, 
                              'camera': camera_times}
    indeces_to_select = index_alignment(x_collection=time_stamps_collection)

    # extract desired indeces for temporal alignment
    ref_move_hand = ref_move_hand[indeces_to_select['hand']]
    ref_move_wrist_FE = ref_move_wrist_FE[indeces_to_select['hand']]
    ref_move_wrist_PS = ref_move_wrist_PS[indeces_to_select['hand']]
    mes_hand = mes_hand[indeces_to_select['hand']]
    mes_wrist_FE = mes_wrist_FE[indeces_to_select['hand']]
    camera = camera[indeces_to_select['camera']]
    episode_lenth = len(indeces_to_select['hand'])
    
    # switch channels and scale all frames
    camera = [resize_image(frame, scaling_factor=scaling_factor) for frame in camera]
    camera = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in camera]
    camera = [cv2.rotate(frame, cv2.ROTATE_180) for frame in camera]

    # append to merged_store    
    if i == 0:
        merged_store['data']['image_in_hand'] = camera
        merged_store['data']['ref_move_hand'] = ref_move_hand.reshape(-1, 1)
        merged_store['data']['ref_move_wrist_FE'] = ref_move_wrist_FE.reshape(-1, 1)
        merged_store['data']['ref_move_wrist_PS'] = ref_move_wrist_PS.reshape(-1, 1)
        merged_store['data']['mes_hand'] = mes_hand.reshape(-1, 1)
        merged_store['data']['mes_wrist_FE'] = mes_wrist_FE.reshape(-1, 1)
    else:
        merged_store['data']['ref_move_hand'].append(ref_move_hand.reshape(-1, 1), axis=0)
        merged_store['data']['ref_move_wrist_FE'].append(ref_move_wrist_FE.reshape(-1, 1), axis=0)
        merged_store['data']['ref_move_wrist_PS'].append(ref_move_wrist_PS.reshape(-1, 1), axis=0)
        merged_store['data']['mes_hand'].append(mes_hand.reshape(-1, 1), axis=0)
        merged_store['data']['mes_wrist_FE'].append(mes_wrist_FE.reshape(-1, 1), axis=0)
        merged_store['data']['image_in_hand'].append(camera, axis=0)

    merged_store['meta']['episode_ends'].append([episode_lenth])
    
    # need to retrieve the attrs before appending, then reassign
    attrs = merged_store.attrs['label']
    attrs.append(store.attrs['label'])
    merged_store.attrs['label'] = attrs
    
    del store
    del ref_move_hand
    del ref_move_wrist_FE
    del ref_move_wrist_PS
    del mes_hand
    del mes_wrist_FE
    del camera
    del episode_lenth

# compute episode ends form list of episode lengths
# NOTE I had put -1 to make the episode ends match with array indeces, but when creating their ReplayBuffer you get an assertion error.
merged_store['meta']['episode_ends'] = np.cumsum(merged_store['meta']['episode_ends']).astype(np.int32) # - 1

# save merged data in new file
#merged_path = os.path.join(merged_dir, merged_name)
#store_merged_data_to_zarr(merged_data, attrs, path=merged_path)

print("=================")
print("Saved merged store at %s" % merged_path)


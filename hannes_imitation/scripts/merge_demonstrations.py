import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation')

from hannes_imitation.common.data_utils import filter_demonstrations, index_alignment, resize_image, store_merged_data_to_zarr
import os
import zarr
import numpy as np
from tqdm import tqdm

# parameters 
demonstration_dir = '/home/calessi-iit.local/Projects/pyHannesAPI/tests/notebooks/dataset/preliminary/' # input directory
merged_dir = '/home/calessi-iit.local/Projects/hannes-imitation/data/preliminary/' # output directory
merged_name = 'merged.zarr' # output filename
scaling_factor = 0.2 # scaling factor of images

# filter demonstrations
demonstration_names = filter_demonstrations(demonstration_dir)

print("Demonstrations retained:", len(demonstration_names['retained']))
print("Demonstrations discarded:", len(demonstration_names['discarded']))

merged_data = {
    'data': 
        {'image_in_hand': [], 
         'ref_move_hand': []}, 
    'meta': {'episode_ends': []}}

for demonstration_name in tqdm(demonstration_names['retained']):
    store = zarr.open(demonstration_dir + demonstration_name)

    # get desired indeces for temporal alignment for hand time stamps and video time stamps
    hand_times = store['time'][:]
    camera_times = store['in-hand_camera_0']['time'][:]
    time_stamps_collection = {'hand': hand_times, 
                              'camera': camera_times}
    indeces_to_select = index_alignment(x_collection=time_stamps_collection)

    # extract action and observations needed, with temporal alignment
    episode_lenth = len(indeces_to_select['hand'])
    action = store['references']['hand'][indeces_to_select['hand']].reshape(-1, 1) # reshape ref_move_hand (T,) -> (T,1)
    camera = store['in-hand_camera_0']['frames'][indeces_to_select['camera']]
    
    # downsize all frames
    camera = [resize_image(frame, scaling_factor=scaling_factor) for frame in camera]

    # add to merged dataset
    merged_data['data']['ref_move_hand'].extend(action) # reshaped
    merged_data['data']['image_in_hand'].extend(camera)
    merged_data['meta']['episode_ends'].append(len(action))

# compute episode ends form list of episode lengths
# NOTE I had put -1 to make the episode ends match with array indeces, but when creating their ReplayBuffer you get an assertion error.
merged_data['meta']['episode_ends'] = np.cumsum(merged_data['meta']['episode_ends']) # TODO - 1

# save merged data in new file
merged_path = os.path.join(merged_dir, merged_name)
store_merged_data_to_zarr(merged_data, path=merged_path)

print("=============")
print("Saved merged store at %s" % merged_path)


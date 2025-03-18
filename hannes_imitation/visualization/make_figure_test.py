import zarr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import time

import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation/')
from hannes_imitation.common import plot_utils

### load store
#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_power_drill_2025-2-20_18-51-22.zarr'

#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_box_of_sugar_2025-2-20_19-8-50.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_box_of_sugar_2025-2-20_19-12-22.zarr' # iros 2025

#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_2_object_power_drill_2025-2-27_15-10-21.zarr'
#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_2_object_power_drill_2025-2-27_15-39-27.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_2_object_power_drill_2025-2-27_15-44-27.zarr' # iros 2025

#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_7_object_enamel-coated_metal_bowl_2025-2-27_19-21-32.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_7_object_enamel-coated_metal_bowl_2025-2-27_19-27-52.zarr' # iros 2025

#parser = argparse.ArgumentParser(description="Make video of test trials.")

# Adding arguments
#parser.add_argument("--test_data", type=str, default=None, help="Directory where to plot the input data.")

# Parsing arguments
#args = parser.parse_args()

# output video dir
local_time = time.localtime()
timestamp = '_%d_%d_%d-%d_%d_%d' % (local_time.tm_year, local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

store = zarr.open(test_data_path, mode='r')

frames_list = store['camera_frames'][:]
mes_hand = store['joints']['hand']['position'][:]
mes_wrist_FE = store['joints']['wrist_FE']['position'][:]
external_frames_list = store['external']['rgb'][:]
action_trajectories = store['policy_actions'][:]
prediction_timestamps = store['prediction_timestamps']
time_axis = store['time'][:] - store['time'][0]

n_predictions, action_horizon, action_dim = action_trajectories.shape

# convert frames from BGR to RGB
frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list]
frames = np.array([cv2.rotate(frame, cv2.ROTATE_180) for frame in frames]) # rotate eye-in-hand frames
external_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in external_frames_list])
actions = action_trajectories.reshape(-1, action_dim) # get all actions executed

# subsample 
# actions are always less than the rest for now
frames_indeces_subsample = np.linspace(start=0, stop=len(frames)-1, num=len(actions)).astype(np.int64) # -1 because linspace includes stop
frames = frames[frames_indeces_subsample]

time_axis_subsample = np.linspace(start=time_axis[0], stop=time_axis[-1], num=len(actions)) # -1 because linspace includes stop

external_frames_indeces_subsample = np.linspace(start=0, stop=len(external_frames)-1, num=len(actions)).astype(np.int64) # -1 because linspace includes stop
external_frames = external_frames[external_frames_indeces_subsample]

mes_indeces_subsample = np.linspace(start=0, stop=len(mes_hand)-1, num=len(actions)).astype(np.int64)
mes_hand = mes_hand[mes_indeces_subsample]
mes_wrist_FE = mes_wrist_FE[mes_indeces_subsample]

# duration and fps
episode_duration = store['time'][-1] - store['time'][0] #toc - tic
video_fps = len(frames) / episode_duration

print("======== subsampled shapes ==========")
print("Camera frames:", frames.shape)
print("External camera frames:", external_frames.shape)
print("Mes hand:", mes_hand.shape)
print("Mes wrist_FE:", mes_wrist_FE.shape)
print("Executed actions:", actions.shape)
print("Time axis:", time_axis_subsample.shape)

print("======== duration/fps ==========")
print("Trial duration: %.2f s" % episode_duration)
print("Video fps: %.2f" % video_fps)

name = 'collection_1_' + 'box_of_sugar_'
name = 'collection_2_' + 'power_drill_'
name = 'collection_7_' + 'enamel-coated_metal_bowl_'

# TWO ACTIONS
fig = plt.figure(figsize=(6, 4))

rows_grid = 3
cols_grid = 4
gs = gridspec.GridSpec(nrows=rows_grid, ncols=cols_grid, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])

# eye-in-hand camera
axes_frames = []
for i in range(cols_grid):
    ax = fig.add_subplot(gs[1, i])
    axes_frames.append(ax)

# external camera
axes_external_frames = []
for i in range(cols_grid):
    ax = fig.add_subplot(gs[0, i])
    axes_external_frames.append(ax) 

ax_encoder = fig.add_subplot(gs[-1, :]) # encoder measurements axis

nrows = 1
ncols = 4
pad_frames = 2
start_i = 1
end_i = -1

frame_indeces = np.linspace(start=0, stop=len(frames)-1, num=nrows*ncols + pad_frames, dtype=int)
#frame_indeces = frame_indeces[0:-1] #frame_indeces[1:-1]
frame_indeces = frame_indeces[start_i:end_i]
for ax, index in zip(axes_frames, frame_indeces):
    ax.imshow(frames[index])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

axes_frames[0].set_ylabel('Eye-in-hand')

ext_frame_indeces = np.linspace(start=0, stop=len(external_frames)-1, num=nrows*ncols + pad_frames, dtype=int)
#ext_frame_indeces = ext_frame_indeces[0:-1] #ext_frame_indeces[1:-1]
ext_frame_indeces = ext_frame_indeces[start_i:end_i] #ext_frame_indeces[1:-1]

for ax, index in zip(axes_external_frames, ext_frame_indeces):
    ax.imshow(external_frames[index])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

axes_external_frames[0].set_ylabel('External')

ax_encoder.plot(time_axis_subsample, mes_hand / 1000, color='tab:red', linewidth=2, alpha=1, label='Hand O/C')[0]
ax_encoder.plot(time_axis_subsample, mes_wrist_FE / 1000, color='tab:green', linewidth=2, alpha=1, label='Wrist F/E')[0]
ax_encoder.set_ylabel('Encoder')
ax_encoder.grid(linewidth=0.5, linestyle='--')
ax_encoder.set_xlabel("Time, $t$ (s)")
ax_encoder.legend(loc='upper left', ncols=1)
ax_encoder.set_yticklabels([])
#import matplotlib.ticker
#ax_encoder.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.0f}e3")) # Add "×10³" near the y-axis label #× 10³

plt.savefig(fname='../../figures/test/iros2025/' + name + 'observations.pdf', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/test/iros2025/' + name + 'observations.png', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/test/iros2025/' + name + 'observations.svg', bbox_inches='tight', dpi=600)
plt.tight_layout()
plt.show()

"""
Plot actions
"""
# Create the top-right plot
fig = plt.figure(figsize=(6, 4))
plt.plot(time_axis_subsample, actions[:, 0], color='tab:red', linewidth=2, alpha=1, label="Hand O/C")[0]
plt.plot(time_axis_subsample, actions[:, 1], color='tab:green', linewidth=2, alpha=1, label='Wrist F/E')[0]
plt.plot(time_axis_subsample, actions[:, 2], color='navy', linestyle='--', linewidth=2, alpha=1, label='Wrist P/S')[0]
plt.fill_between(x=time_axis_subsample, y1=actions[:, 2], color='navy', linewidth=2, alpha=0.3, label='P/S angular displacement') # NOTE [0] raises plt error (non subscriptable)
plt.ylabel('Action')
plt.xlabel("Time, $t$ (s)")
plt.grid(linewidth=0.5, linestyle='--')
plt.legend(loc='upper left', ncols=1)

plt.savefig(fname='../../figures/test/iros2025/' + name + 'action_predictions.pdf', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/test/iros2025/' + name + 'action_predictions.png', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/test/iros2025/' + name + 'action_predictions.svg', bbox_inches='tight', dpi=600)

# Adjust layout for clarity
plt.tight_layout()
plt.show()
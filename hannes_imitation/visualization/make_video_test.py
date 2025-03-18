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
#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-1-24_15-43-56.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-1-27_17-37-14.zarr'

test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-1-28_17-2-52.zarr' # boccardo 1
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-1-28_17-6-9.zarr' # boccardo 2
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-1-28_17-29-12.zarr' # boccardo 3

# 15 ycb train
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_17-35-7.zarr' # Medium stacking cup orange​
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_17-42-44.zarr' # Enamel-coated metal bowl​
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_17-56-56.zarr' # Flat-head screwdriver​ (fail)
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-5-27.zarr' # power drill
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-17-4.zarr' # box of sugar
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-27-45.zarr' # pear
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-34-0.zarr' # box of gelatine
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-42-48.zarr' # Box of chocolate pudding​
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-47-34.zarr' # Apple
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-53-41.zarr' # Bleach cleanser 1
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-3_18-56-17.zarr' # bleach cleanser 2
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_14-21-46.zarr' # small stacking cup lightblue
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_14-32-36.zarr' # orange
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_14-54-7.zarr' # pitcher
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_15-4-19.zarr' # tennis ball
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_15-45-17.zarr' # foam brick

# 5 ycb test
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_16-40-3.zarr' # container of mustard
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_16-48-51.zarr' # banana
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_17-39-32.zarr' # metal mug
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_17-43-54.zarr' # baseball
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-2-4_17-51-25.zarr' # can of potted meat

# quick trial policy 1_4
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_14-45-20.zarr' # sugar table
#test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_14-48-9.zarr' # sugar shelf
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_15-0-52.zarr' # water blue shelf
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_15-4-36.zarr' # water blue shelf clutter
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_15-10-43.zarr' # water red table clutter
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_15-14-47.zarr' # water neutral table clutter
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_15-22-19.zarr' # water green table clutter
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1_4/trial_2025-2-11_15-30-57.zarr' # water neutral small table clutter

test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1-8/collection_3_object_blue_bottle_05l_2025-2-18_18-24-5.zarr'

test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_power_drill_2025-2-20_18-51-22.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_box_of_sugar_2025-2-20_19-8-50.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_box_of_sugar_2025-2-20_19-12-22.zarr'


test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_box_of_sugar_2025-2-20_19-12-22.zarr' # iros 2025
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_2_object_power_drill_2025-2-27_15-44-27.zarr' # iros 2025
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_7_object_enamel-coated_metal_bowl_2025-2-27_19-27-52.zarr' # iros 2025

#parser = argparse.ArgumentParser(description="Make video of test trials.")

# Adding arguments
#parser.add_argument("--test_data", type=str, default=None, help="Directory where to plot the input data.")
#parser.add_argument("--video_path", type=str, default=None, help="Directory where to store the output video.")

# Parsing arguments
#args = parser.parse_args()

# output video dir
name = 'collection_1_' + 'box_of_sugar_'
name = 'collection_2_' + 'power_drill_'
name = 'collection_7_' + 'enamel-coated_metal_bowl_'

local_time = time.localtime()
timestamp = '_%d_%d_%d-%d_%d_%d' % (local_time.tm_year, local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)
video_path = "../../figures/iros2025/test/" + name + timestamp + ".mp4"

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

# TWO ACTIONS
fig = plt.figure(figsize=(12, 4)) # (6,4) for half fig in paper
#fig = plt.figure(figsize=(16, 9))

rows_grid = 3
cols_grid = 5
gs = gridspec.GridSpec(nrows=rows_grid, ncols=cols_grid, width_ratios=[1, 1, 1, 1, 4], height_ratios=[1, 1, 1])

ax_ext_frames = fig.add_subplot(gs[:2, :2])
ax_frames = fig.add_subplot(gs[:2, 2:4])
ax_encoder = fig.add_subplot(gs[2, :4])
ax_action = fig.add_subplot(gs[:, 4])

# external view plot
line_ext_img = ax_ext_frames.imshow(external_frames[0])
ax_ext_frames.set_xticks([])
ax_ext_frames.set_yticks([])
ax_ext_frames.set_title('External view')

# eye-in-hand plot
line_img = ax_frames.imshow(frames[0])
ax_frames.set_xticks([])
ax_frames.set_yticks([])
ax_frames.set_title('Eye-in-hand')

# action plot
line_action_pred_1 = ax_action.plot(time_axis_subsample, actions[:, 0], color='tab:red', linewidth=2, alpha=1, label="Hand O/C")[0]
line_action_pred_2 = ax_action.plot(time_axis_subsample, actions[:, 1], color='tab:green', linewidth=2, alpha=1, label='Wrist F/E')[0]
line_action_pred_3 = ax_action.plot(time_axis_subsample, actions[:, 2], color='navy', linewidth=2, alpha=1, label='Wrist P/S')[0]
ax_action.fill_between(x=time_axis_subsample, y1=actions[:, 2], color='navy', linewidth=2, alpha=0.3, label='P/S angular displacement') # NOTE [0] raises plt error (non subscriptable)
ax_action.set_ylabel('Action')
ax_action.grid(linewidth=0.5, linestyle='--')
ax_action.set_xlabel("Time, $t$ (s)")
ax_action.legend(loc='upper left', ncols=1)

# encoder plot
line_mes_1 = ax_encoder.plot(time_axis_subsample, mes_hand, color='tab:red', linewidth=2, alpha=1, label='Hand O/C')[0]
line_mes_2 = ax_encoder.plot(time_axis_subsample, mes_wrist_FE, color='tab:green', linewidth=2, alpha=1, label='Wrist F/E')[0]
ax_encoder.set_ylabel('Encoder')
ax_encoder.grid(linewidth=0.5, linestyle='--')
ax_encoder.set_xlabel("Time, $t$ (s)")
ax_encoder.legend(loc='upper left', ncols=1)
ax_encoder.set_yticklabels([])

# Adjust layout for clarity
plt.tight_layout()

# Initialize function
def init():
    line_ext_img.set_data([[]])
    line_img.set_data([[]])
    line_action_pred_1.set_data([], [])
    line_action_pred_2.set_data([], [])
    line_action_pred_3.set_data([], [])
    line_mes_1.set_data([], [])
    line_mes_2.set_data([], [])

    return line_ext_img, line_img, line_action_pred_1, line_action_pred_2, line_action_pred_3, line_mes_1, line_mes_2

# Update function
def update(i):
    
    line_ext_img.set_data(external_frames[i])
    line_img.set_data(frames[i])

    ax_action.collections.clear()
    line_action_pred_1.set_data(time_axis_subsample[:i], actions[:i, 0])
    line_action_pred_2.set_data(time_axis_subsample[:i], actions[:i, 1])
    line_action_pred_3.set_data(time_axis_subsample[:i], actions[:i, 2])
    ax_action.fill_between(x=time_axis_subsample[:i], y1=actions[:i, 2], color='navy', linewidth=2, alpha=0.3, label='P/S angular displacement') # NOTE [0] raises plt error (non subscriptable)

    line_mes_1.set_data(time_axis_subsample[:i], mes_hand[:i])
    line_mes_2.set_data(time_axis_subsample[:i], mes_wrist_FE[:i])

    return line_ext_img, line_img, line_action_pred_1, line_action_pred_2, line_action_pred_3, line_mes_1, line_mes_2

# Create animation
ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False)
ani.save(video_path, writer="ffmpeg", fps=video_fps, dpi=600)

plt.show()
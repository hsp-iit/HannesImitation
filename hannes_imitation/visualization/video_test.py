import zarr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import numpy as np
import cv2

import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation')
from hannes_imitation.common import plot_utils

### load store
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/trial_2025-1-24_15-43-56.zarr'

store = zarr.open(test_data_path, mode='r')

frames_list = store['camera_frames'][:]
action_trajectories = store['policy_actions'][:]
prediction_timestamps = store['prediction_timestamps']
time_axis = store['time'][:] - store['time'][0]

n_predictions, action_horizon, action_dim = action_trajectories.shape

# convert frames from BGR to RGB
frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list])
actions = action_trajectories.reshape(-1, action_dim) # get all actions executed

# subsample 
# actions are always less than the rest for now
frames_indeces_subsample = np.linspace(start=0, stop=len(frames)-1, num=len(actions)).astype(np.int64) # -1 because linspace includes stop
frames = frames[frames_indeces_subsample]

time_axis_subsample = np.linspace(start=time_axis[0], stop=time_axis[-1], num=len(actions)) # -1 because linspace includes stop

# duration and fps
episode_duration = store['time'][-1] - store['time'][0] #toc - tic
video_fps = len(frames) / episode_duration

print("Trial duration: %.2f s" % episode_duration)
print("Video fps: %.2f" % video_fps)

# TWO ACTIONS
fig = plt.figure(figsize=(10, 6))

gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[2, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[:, 0])  # Span all rows in the first column
ax2 = fig.add_subplot(gs[0, 1])  # Top-right
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)  # Bottom-right

# Create the left plot (spanning both rows), images
line_img = ax1.imshow(frames[0])
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Eye-in-hand camera view', fontweight="bold")

# Create the top-right plot
line_action_pred_1 = ax2.plot(time_axis_subsample, actions[:, 0], color='blue', linewidth=2, alpha=0.8, label='Policy prediction')[0]
ax2.set_ylabel("Hand\n Open $-$ Close")
ax2.grid(linewidth=0.5, linestyle='--')
ax2.legend(loc='upper left', ncols=1)

# Create the bottom-right plot
line_action_pred_2 = ax3.plot(time_axis_subsample, actions[:, 1], color='blue', linewidth=2, alpha=0.8, label='Prediction')[0]
ax3.set_ylabel('Wrist\n Flex $-$ Extend')
ax3.grid(linewidth=0.5, linestyle='--')
ax3.set_xlabel("Time, $t$ (s)")

# Adjust layout for clarity
plt.tight_layout()

# Initialize function
def init():
    line_img.set_data([[]])
    line_action_pred_1.set_data([], [])
    line_action_pred_2.set_data([], [])

    return line_img, line_action_pred_1, line_action_pred_2

# Update function
def update(i):
    line_img.set_data(frames[i])

    line_action_pred_1.set_data(time_axis_subsample[:i], actions[:i, 0])
    line_action_pred_2.set_data(time_axis_subsample[:i], actions[:i, 1])

    return line_img, line_action_pred_1, line_action_pred_2

# Create animation
ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False)
ani.save("../../figures/hannes_policy_evaluation_hand_wrist_FE_online-tmp-x.mp4", writer="ffmpeg", fps=video_fps, dpi=100)

plt.show()
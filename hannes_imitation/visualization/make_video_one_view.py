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

test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_1_object_box_of_sugar_2025-2-20_19-12-22.zarr' # iros 2025
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_2_object_power_drill_2025-2-27_15-44-27.zarr' # iros 2025
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/collection_7_object_enamel-coated_metal_bowl_2025-2-27_19-27-52.zarr' # iros 2025


test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/other/collection_1_object_mug_2025-3-5_16-51-47.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/other/collection_1_object_banana_2025-3-5_16-55-51.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/other/collection_2_object_mustard_2025-3-5_17-4-15.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/other/collection_2_object_mustard_2025-3-5_16-59-49.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/other/collection_2_object_meat_2025-3-5_17-7-5.zarr'
test_data_path = '/home/calessi-iit.local/Projects/hannes-imitation/data/test/iros2025/other/collection_3_object_baseball_2025-3-5_16-32-36.zarr'


# Parsing arguments
#args = parser.parse_args()

# output video dir
#name = 'collection_1_' + 'box_of_sugar_'
#name = 'collection_2_' + 'power_drill_'
#name = 'collection_7_' + 'enamel-coated_metal_bowl_'

view_1 = 'external_'
view_2 = 'palm_'

obj = 'mug_'
obj = 'banana_'
obj = 'mustard_'
obj = 'meat_'
obj = 'baseball_'

collection = 'collection_%d_' % 3

name_1 = view_1 + collection + obj
name_2 = view_2 + collection + obj


local_time = time.localtime()
timestamp = '_%d_%d_%d-%d_%d_%d' % (local_time.tm_year, local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)
video_path_1 = "../../figures/iros2025/test/other/" + name_1 + timestamp + ".mp4"
video_path_2 = "../../figures/iros2025/test/other/" + name_2 + timestamp + ".mp4"

store = zarr.open(test_data_path, mode='r')

frames_list = store['camera_frames'][:]
external_frames_list = store['external']['rgb'][:]
action_trajectories = store['policy_actions'][:]
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

# duration and fps
episode_duration = store['time'][-1] - store['time'][0] #toc - tic
video_fps = len(frames) / episode_duration

print("======== subsampled shapes ==========")
print("Camera frames:", frames.shape)
print("External camera frames:", external_frames.shape)
print("Executed actions:", actions.shape)
print("Time axis:", time_axis_subsample.shape)

print("======== duration/fps ==========")
print("Trial duration: %.2f s" % episode_duration)
print("Video fps: %.2f" % video_fps)


def make_ext_view():
    print("make_ext_view")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    height = external_frames.shape[1]
    width = external_frames.shape[2]
    out = cv2.VideoWriter(video_path_1, fourcc, video_fps, (width, height))

    # Loop through images and write them to the video
    for image in external_frames:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)  # Add frame to video

    # Release the video writer
    out.release()
    #return

    #fig_ext, ax = plt.subplots(figsize=(16, 9))

    # external view plot
    #line_ext_img = ax.imshow(external_frames[0])
    #ax.set_xticks([])
    #ax.set_yticks([])
    #plt.tight_layout(pad=0)  # Remove padding
    #plt.margins(0)

    # Initialize function
    #def init():
    #    line_ext_img.set_data([[]])

    #    return line_ext_img

    # Update function
    #def update(i):
    #    line_ext_img.set_data(external_frames[i])

    #    return line_ext_img

    # Create animation
    #ani = FuncAnimation(fig_ext, update, frames=len(frames), init_func=init, blit=False)
    #ani.save(video_path_1, writer="ffmpeg", fps=video_fps, dpi=100)

    #plt.show()
    #plt.close(fig_ext)

"""
PALM
"""
def make_palm_view():
    print("make_palm_view")
    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    height = frames.shape[1]
    width = frames.shape[2]
    out = cv2.VideoWriter(video_path_2, fourcc, video_fps, (width, height))

    # Loop through images and write them to the video
    for image in frames:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)  # Add frame to video

    # Release the video writer
    out.release()
    #return

    #fig_palm, ax = plt.subplots(figsize=(4, 3))
    #line_img = ax.imshow(frames[0].astype(np.int8), aspect='auto')
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.axis("off")  # Remove axis completely
    #ax.set_position([0, 0, 1, 1])  # Expand axes to fill figure
    #plt.tight_layout(pad=0)  # Remove padding

    # Initialize function
    #def init_palm():
    #    line_img.set_data([[]])

    #    return line_img

    # Update function
    #def update_palm(i):
    #    line_img.set_data(frames[i])

    #    return line_img

    # Create animation
    #ani = FuncAnimation(fig_palm, update_palm, frames=len(frames), init_func=init_palm, blit=False)
    #ani.save(video_path_2, writer="ffmpeg", fps=video_fps, dpi=600)
    

    #plt.show()
    #plt.close(fig_palm)



if __name__ == '__main__':
    make_ext_view()
    make_palm_view()
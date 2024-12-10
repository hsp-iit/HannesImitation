import numpy as np
import matplotlib.pyplot as plt

# hannnes_imitation import
import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation')
from hannes_imitation.common.data_utils import index_alignment


# time stamps of data source 1
final_time_1 = 5 # s
n_sample_1 = 100
delta_t_1 = final_time_1 / n_sample_1 # s
timestamps_noise_1 = np.random.uniform(low=delta_t_1/3, high=delta_t_1/2, size=n_sample_1)
timestamps_1 = np.linspace(start=0, stop=final_time_1, num=n_sample_1)
timestamps_1 += timestamps_noise_1

# time stamps of data source 2
final_time_2 = final_time_1 + np.random.normal(loc=0, scale=0.1) # s
n_sample_2 = 230
delta_t_2 = final_time_2 / n_sample_2 #
timestamps_noise_2 = np.random.uniform(low=-delta_t_2/3, high=delta_t_2/3, size=n_sample_2)
timestamps_2 = np.linspace(start=0, stop=final_time_2, num=n_sample_2)
timestamps_2 += timestamps_noise_2


time_stamps_collection = {'timestamps_1': timestamps_1, 
                          'timestamps_2': timestamps_2}

indeces_to_select = index_alignment(x_collection=time_stamps_collection)

new_timestamps_1 = timestamps_1[indeces_to_select['timestamps_1']]
new_timestamps_2 = timestamps_2[indeces_to_select['timestamps_2']]


# visualize
plt.figure(figsize=(9,4))

# plot raw time stamps
plt.plot(timestamps_1, np.ones_like(timestamps_1), 'o', color='blue')
plt.plot(timestamps_2, np.zeros_like(timestamps_2), 'o', color='red')

# plot new time stamps
plt.plot(new_timestamps_1, np.ones_like(new_timestamps_1), 'o', color='blue')
plt.plot(new_timestamps_2, np.zeros_like(new_timestamps_2), 'o', color='red')

# draw connecting lines
for hand_time, camera_time in  zip(new_timestamps_1, new_timestamps_2):
    plt.plot([camera_time, hand_time], [0, 1], '--', color='black')

plt.grid(axis='x')
plt.yticks([0, 1], ['Camera',  'Hand'])
plt.xlabel("Timestamp (s)")
plt.xlim([timestamps_1[0]-0.05, timestamps_1[30]+0.05])
plt.show()


# print average and maximum temporal disalignment
aligned_timestamps_diffs = np.abs(new_timestamps_1 - new_timestamps_2)

print("max time disalignment: %.3f s" % np.max(aligned_timestamps_diffs)) 
print("avg time disaligmnet: %.3f s " % np.mean(aligned_timestamps_diffs))
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../hannes-imitation/')
from hannes_imitation.common import plot_utils

# Generate Gaussian noise
prediction_horizon = 8
n_actions = 3
noise = np.random.normal(loc=0, scale=1, size=(prediction_horizon, n_actions))

# Plot the noise as an image
plt.figure(figsize=(3, 8))  # Set figure size to match 8x3 aspect ratio
plt.imshow(noise, cmap='gray', aspect='auto')
#plt.colorbar(label="Intensity")
#plt.title("8x3\nRandom\nGaussian Noise")
plt.axis('off')  # Hide axis for better visualization
plt.tight_layout()

plt.savefig(fname='../../figures/iros2025/gaussian_noise_sample.pdf', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/iros2025/gaussian_noise_sample.png', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/iros2025/gaussian_noise_sample.svg', bbox_inches='tight', dpi=600)
plt.show()

# Generate independent Gaussian noise for each color channel
n_channels = 3
denoised_action = np.ones((prediction_horizon, n_actions, n_channels)) * 0.2

denoised_action[:, 0, 0] = np.linspace(0.5, 0.8, 8) # first column is red
denoised_action[:, 1, 1] = np.linspace(0.9, 0.3, 8) # second column is green
denoised_action[:4, 2, 2] = np.linspace(0.5, 1, 4) # third column is blue
denoised_action[4:, 2, 2] = 0.5 # third column is blue

# Display the image
#plt.figure(figsize=(3, 8))
f = plt.figure(figsize=(3, 8))
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.imshow(denoised_action, aspect='auto')
#plt.title("8x3\nDenoised Hand & Wrist\nAction trajectory")
ax.set_yticks(np.arange(prediction_horizon), ['$t$' if i==0 else '$t+%d$' % i  for i in range(prediction_horizon)], fontsize=28)
ax.set_xticks([])
plt.tight_layout()

plt.savefig(fname='../../figures/iros2025/denoised_action_trajectory.pdf', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/iros2025/denoised_action_trajectory.png', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/iros2025/denoised_action_trajectory.svg', bbox_inches='tight', dpi=600)
plt.show()


# Display the image
#plt.figure(figsize=(3, 8))
action_horizon = 4
f = plt.figure(figsize=(3, action_horizon))
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.imshow(denoised_action[:action_horizon], aspect='auto')
#plt.title("8x3\nDenoised Hand & Wrist\nAction trajectory")
ax.set_yticks(np.arange(action_horizon), ['$t$' if i==0 else '$t+%d$' % i  for i in range(action_horizon)], fontsize=28)
ax.set_xticks([])
plt.tight_layout()

plt.savefig(fname='../../figures/iros2025/executed_actions.pdf', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/iros2025/executed_actions.png', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/iros2025/executed_actions.svg', bbox_inches='tight', dpi=600)
plt.show()
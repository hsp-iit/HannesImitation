"""
Takes the results of validate_policy.py and creates distribution of action errors in the form of violin/box plots.
The distributions are shown grouping the errors by scenarios and then by action.
It also prints summary errors.
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/calessi-iit.local/Projects/hannes-imitation/')
from hannes_imitation.common import plot_utils


data = np.load('/home/calessi-iit.local/Projects/hannes-imitation/data/validation/validation_set_results.npz', allow_pickle=True)
results_dicts = [data[key].item() for key in data.files]
action_horizon = 4

# process results
table_grasp_errors = []
shelf_grasp_errors = []
human_to_hannes_handover_errors = []

for result_dict in results_dicts:
    for h in range(action_horizon):
        errors = result_dict['errors_%d' % h] # these are diffs

        if result_dict['scenario'] == 'table':
            table_grasp_errors.extend(errors)
        if result_dict['scenario'] == 'shelf':
            shelf_grasp_errors.extend(errors)
        if result_dict['scenario'] == '-':
            human_to_hannes_handover_errors.extend(errors)

table_grasp_errors = np.array(table_grasp_errors) 
shelf_grasp_errors = np.array(shelf_grasp_errors) 
human_to_hannes_handover_errors = np.array(human_to_hannes_handover_errors) 

table_grasp_errors = np.abs(table_grasp_errors) / 100 * 100
shelf_grasp_errors = np.abs(shelf_grasp_errors) / 100 * 100
human_to_hannes_handover_errors = np.abs(human_to_hannes_handover_errors) / 60 * 100

scenario_labels = ['Table\nGrasp', 'Shelf\nGrasp', 'Human-to-Hannes\nHandover']
action_labels = ['Hand O/C', 'Wrist F/E', 'Wrist P/S']
violin_colors = ['tab:red', 'tab:green', 'navy']
error_data = [table_grasp_errors, shelf_grasp_errors, human_to_hannes_handover_errors]

# print summary
print("===== overall error ====")
print("N. samples: %d" % table_grasp_errors.shape[0])
print("Overall error: %.1f +- %.1f" % (np.concatenate(error_data).mean(), np.concatenate(error_data).std()))
print()

print("===== error by scenario ====")
for i, error in enumerate(error_data):
    print(scenario_labels[i].replace("\n", " "), "Average error: %.1f +- %.1f" % (np.mean(error), np.std(error)))
print()

print("===== error by action ====")
mean_error_by_action = np.concatenate(error_data).mean(axis=0)
std_error_by_action = np.concatenate(error_data).std(axis=0)
for i, action_label in enumerate(action_labels):
    print("%s: %.1f +- %.1f" % (action_label, mean_error_by_action[i], std_error_by_action[i]))
print()

print("===== error by scenario by action ====")
for i, error in enumerate(error_data):
    print(scenario_labels[i].replace("\n", " "), np.mean(error, axis=0).round(1), "+-", np.std(error, axis=0).round(1))
print()


 
# Setup figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharey=True)

for i, ax in enumerate(axes):
    violin_parts = ax.violinplot(error_data[i], showmeans=False, showmedians=False, showextrema=False)

    # Style violin plots
    for j, body in enumerate(violin_parts['bodies']):
        body.set_facecolor(violin_colors[j])
        body.set_edgecolor(violin_colors[j])
        body.set_alpha(0.5)
        body.set_linewidth(2)

    # Add boxplots on top
    positions = np.arange(1, error_data[i].shape[1] + 1)
    box_parts = ax.boxplot(
        error_data[i],
        positions=positions,
        showfliers=False,
        widths=0.2,
        capwidths=0.2,
        patch_artist=True,
        boxprops=dict(facecolor='white', color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5))

    # Style medians individually
    for j, median in enumerate(box_parts['medians']):
        median.set_color(violin_colors[j])
        median.set_linewidth(1.5)
        
    ax.grid(linewidth=0.5, linestyle='--')
    ax.set_xticklabels([])
    #ax.set_xlabel(scenario_labels[i])
    ax.set_title(scenario_labels[i], fontsize=12)
    ax.set_facecolor('lightgrey')

axes[0].set_ylabel('Absolute action error (%)')

# legends
axes[0].plot([], [], color=violin_colors[0], label=action_labels[0])
axes[0].plot([], [], color=violin_colors[1], label=action_labels[1])
axes[0].plot([], [], color=violin_colors[2], label=action_labels[2])
axes[0].legend(loc='upper left')

plt.tight_layout()

plt.savefig(fname='../../figures/validation/validation_error_distributions.pdf', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/validation/validation_error_distributions.png', bbox_inches='tight', dpi=600)
plt.savefig(fname='../../figures/validation/validation_error_distributions.svg', bbox_inches='tight', dpi=600)
plt.show()


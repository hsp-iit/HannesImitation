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

# process results
table_grasp_errors = []
shelf_grasp_errors = []
human_to_hannes_handover_errors = []

for result_dict in results_dicts:
    if result_dict['scenario'] == 'table':
        table_grasp_errors.extend(result_dict['errors'])
    if result_dict['scenario'] == 'shelf':
        shelf_grasp_errors.extend(result_dict['errors'])
    if result_dict['scenario'] == '-':
        human_to_hannes_handover_errors.extend(result_dict['errors'])

table_grasp_errors = np.array(table_grasp_errors)
shelf_grasp_errors = np.array(shelf_grasp_errors)
human_to_hannes_handover_errors = np.array(human_to_hannes_handover_errors)

table_grasp_errors = np.abs(table_grasp_errors)
shelf_grasp_errors = np.abs(shelf_grasp_errors)
human_to_hannes_handover_errors = np.abs(human_to_hannes_handover_errors)

scenario_labels = ['Table\nGrasp (#1)', 'Shelf\nGrasp (#2)', 'Human-to-Hannes\nHandover (#3)']
action_labels = ['Hand O/C', 'Wrist F/E', 'Wrist P/S']
violin_colors = ['tab:red', 'tab:green', 'navy']
error_data = [table_grasp_errors, shelf_grasp_errors, human_to_hannes_handover_errors]

 
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
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor='white', color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5))

    # Style medians individually
    for j, median in enumerate(box_parts['medians']):
        median.set_color(violin_colors[j])
        median.set_linewidth(1.5)
        
    ax.grid(linewidth=0.5, linestyle='--', axis='y')
    ax.set_xticks([])
    ax.set_xlabel(scenario_labels[i])
    ax.set_facecolor('lightgrey')

axes[0].set_ylabel('Action error')

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

# print summary
for i, error in enumerate(error_data):
    print(scenario_labels[i].replace("\n", " "))
    print("Average error: %.1f +- %.1f" % (np.mean(error), np.std(error)))
    print("Average error by action:", np.mean(error, axis=0).round(1))
    print("Maximum error by action:", np.max(error, axis=0).round(1))
    print()
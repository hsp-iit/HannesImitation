import matplotlib.pyplot as plt

FONT_SIZE   = 12
TICK_SIZE   = 12
LEGEND_SIZE = 12
AXES_SIZE   = 12

# change font sizes
plt.rc('font',   size=FONT_SIZE)           # controls default text sizes
plt.rc('axes',   labelsize=AXES_SIZE)      # font size of the x and y labels
plt.rc('xtick',  labelsize=TICK_SIZE)      # font size of the tick labels
plt.rc('ytick',  labelsize=TICK_SIZE)      # font size of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)     # legend font size

# removes Type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
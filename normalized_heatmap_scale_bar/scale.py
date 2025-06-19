import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Create a figure and axis with appropriate size
fig, ax = plt.subplots(figsize=(6, 1))  # 6 inches wide, 1 inch tall
fig.subplots_adjust(bottom=0.5)  # Adjust bottom margin

# Define the colormap and normalization from 0 to 1
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# Create a horizontal colorbar
cb1 = mpl.colorbar.ColorbarBase(ax,
                                cmap=cmap,
                                norm=norm,
                                orientation='horizontal')

# Add a label (optional)
cb1.set_label('Scale (0 to 1)')

# Save the figure as an SVG file
plt.savefig("viridis_scale_bar.svg", format='svg', bbox_inches='tight')

# Optional: show the plot if running interactively
plt.show()

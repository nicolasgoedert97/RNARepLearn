import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_bpp(bpp, title=None):
    ax = sns.heatmap(bpp,cmap="hot_r")
    ax.title.set_text(title)
    plt.show()

def compare_bpps(bpp1, bpp2):
    fig, ax = plt.subplots(1, 2,figsize=(15, 5), sharex=True, sharey=True)

    cbar_ax = fig.add_axes([.91, .3, .01, .4])

    sns.heatmap(bpp1,cmap="hot_r",ax=ax[0], cbar_ax=cbar_ax )

    sns.heatmap(bpp2,cmap="hot_r",ax=ax[1], cbar_ax=cbar_ax)
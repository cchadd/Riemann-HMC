#%%
import numpy as np
import matplotlib.pyplot as plt
from hmc import HMC
from scipy.signal import correlate
from matplotlib import cm
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


def cov_ellipse(mean, cov, ax, N=4, data='truth'):
    """
    Function defined to plot the confidence ellipsoids
    """ 

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    v_x, v_y = eigvecs[:, 0][0], eigvecs[:, 0][1]
    alpha = np.arctan2(v_y , v_x)
    width, height = 2 * np.sqrt(eigvals)

    if data == 'truth':
        for i in range(N):
            ax.add_patch(Ellipse(mean, i * width, i * height, lw=1, angle=np.degrees(alpha),
                             ec='g', fc='none', linestyle='-'))
    elif data == 'simulated':
        for i in range(N):
            ax.add_patch(Ellipse(mean.T, i * width, i * height, lw=1, angle=np.degrees(alpha),
                             ec='brown', fc='none', linestyle='-'))

    elif data == 'start':
        for i in range(N):
            ax.add_patch(Ellipse(mean.T, i * width, i * height, lw=1, angle=np.degrees(alpha),
                             ec='green', fc='none', linestyle='-'))

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_array, issparse

def hinton(A, ax=None, marker='s'): 
    """Draw Hinton diagram for visualizing a weight matrix."""
    A = A if issparse(A) else coo_array(A)

    i,j = A.coords[1]+0.5, A.coords[0]+0.5
    ax = ax if ax is not None else plt.gca()
    
    ax.patch.set_facecolor('gray')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_aspect('equal')
    ax.set_xlim(-0.5,A.shape[1]+0.5)
    ax.set_ylim(-0.5, A.shape[0]+0.5)
    plt.gcf().canvas.draw()

    ext = ax.get_window_extent()
    base_size = ((ext.width)*(72./plt.gcf().dpi)/(A.shape[1]+2))**2
    size = np.abs(A.data)
    size = base_size*size/size.max()
    color=np.where(A.data>0, 'white', 'black')

    ax.scatter(
        i,j,
        s=size,
        c=color,
        marker=marker,
        linewidth=0,
    )
    ax.autoscale_view()
    ax.invert_yaxis()

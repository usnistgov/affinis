import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from jaxtyping import Num
from scipy.sparse import coo_array, issparse,sparray

def hinton(A:Num[np.ndarray|sparray, "a b"], ax:Axes|None=None, marker:str='s'): 
    """Draw Hinton diagram for visualizing a 2-D matrix.
    
    This has a significant speed-up over the looping version from the
    [matplotlib cookbook](https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html),
    since we utilize the `Axes.scatter` function, along with some clever
    math to determine marker scaling to fill the canvas correctly. 

    Args:
      A: 2-D matrix to visualize
      ax: Optional pre-defined axis to plot on
      marker: Any valid [`matplotlib.markers`](https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers) option.

    """
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

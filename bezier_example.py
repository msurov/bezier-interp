import numpy as np
import matplotlib.pyplot as plt
from bezier import interpolate, eval_bezier
from mpl_toolkits.mplot3d import Axes3D


def example():
    pts = np.array([
        [0., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 1., 1.],
        [0., 0., 2.],
        [0., 1., 2.],
        [1., 1., 2.],
        [0., 0., 3.],
        [0., 1., 3.],
        [1., 1., 3.],
    ])
    bezier = interpolate(pts, 5, 
        derivs_left=[(1, [-1., 0., 1.])],
        derivs_right=[(1, [0., -1., 1.])]
    )
    t = np.linspace(0, 1, 20)
    vals = np.concatenate([eval_bezier(c, t) for c in bezier], axis=0)
    plt.gcf().add_subplot(111, projection='3d')
    plt.plot(pts[:,0], pts[:,1], pts[:,2], 'o')
    plt.plot(vals[:,0], vals[:,1], vals[:,2])
    plt.show()


if __name__ == '__main__':
    example()

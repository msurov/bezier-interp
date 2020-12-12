A Python3 library for Bezier interpolation of N-dimensional curve

```python
import numpy as np
import matplotlib.pyplot as plt
from bezier import interpolate, eval_bezier
from mpl_toolkits.mplot3d import Axes3D

points = [
  [0., 0., 0.],
  [0., 1., 1.],
  [1., 1., 2.],
  [1., 0., 3.]
]
bezier = interpolate(points, order=5)
t = np.linspace(0, 1, 20)
vals = np.concatenate([eval_bezier(c, t) for c in bezier], axis=0)
plt.plot(pts[:,0], pts[:,1], pts[:,2], 'o')
plt.plot(vals[:,0], vals[:,1], vals[:,2])
plt.show()
```

The function `interpolate` allows define boundary constraint for derivatives as well as optimality criteria
(minimize integral of square of second derivative of the curve).

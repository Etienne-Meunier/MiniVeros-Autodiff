## Error with numpy / jax 



Sometimes we don't manage to reproduce exactly veros results because veros has numpy / jax conversion which gives slightly different results :

```python
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp

pi = np.pi
yt = np.array([-45., -43., -41., -39., -37., -35., -33., -31., -29., -27., -25.,
               -23., -21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,-3., 								 -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19., 21.,  23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  39.,  41.,
                  43.,  45.])

ref = jnp.asarray(yt * pi / 180.0)              # numpy compute, cast after
jax_result = jnp.asarray(yt) * pi / 180.0        # jax compute directly

diff = jax_result - ref
print("max abs diff:", jnp.max(jnp.abs(diff)))
print(diff)
```

For instance here, some of the results has a gap with the two versions : 



```bash
(diffusion) ➜  MiniVeros-Autodiff git:(main) ✗ python test_fastmath.py
max abs diff: 1.1102230246251565e-16
[ 0.00000000e+00 -1.11022302e-16 -1.11022302e-16  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -5.55111512e-17 -5.55111512e-17
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -6.93889390e-18  0.00000000e+00  0.00000000e+00
  6.93889390e-18  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.55111512e-17  5.55111512e-17
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  1.11022302e-16
  1.11022302e-16  0.00000000e+00]
```


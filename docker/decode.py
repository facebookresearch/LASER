import numpy as np
import sys

dim = 1024
X = np.fromfile(sys.argv[1], dtype=np.float32, count=-1)
X.resize(X.shape[0] // dim, dim)
print(X)

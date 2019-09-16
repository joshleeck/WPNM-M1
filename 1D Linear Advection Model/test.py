
import numpy as np


a = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
print(a)

b = np.reshape(a, (10,1))
print(b)
print(np.reshape(b, (2,5)))
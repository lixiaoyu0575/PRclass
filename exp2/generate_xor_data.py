import numpy as np
from sklearn import preprocessing
import random
import math
import matplotlib.pyplot as plt
r = 0.75
print
def generate_xor_data(center, max_r, tag, num):
    points = np.array([center[0], center[1], tag])
    for i in range(num):
        theta = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, max_r)
        x = r * math.sin(theta) + center[0]
        y = r * math.cos(theta) + center[1]
        points = np.row_stack((points, [x, y, tag]))
    return points
class0_data1 = generate_xor_data([0, 0], r, 0.25, 1000)
class0_data2 = generate_xor_data([1, 1], r, 0.25, 1000)
class0_data = np.concatenate((class0_data1, class0_data2))
class1_data1 = generate_xor_data([0, 1], r, 0.75, 1000)
class1_data2 = generate_xor_data([1, 0], r, 0.75, 1000)
class1_data = np.concatenate((class1_data1, class1_data2))
generated_xor_data = np.concatenate((class0_data, class1_data))
plt.plot(class0_data[:, 0], class0_data[:, 1], '+')
plt.plot(class1_data[:, 0], class1_data[:, 1], '+')
plt.show()
print class0_data
np.savetxt("generated_XOR_data.txt", generated_xor_data)

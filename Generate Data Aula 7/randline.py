#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

n= 1000

x = np.random.rand(n)
y = np.random.rand(n)

a , b = np.polyfit(x, y, 1)
plt.ylim(-10 ,10)
plt.xlim(-5 , 5)
plt.plot(x, a*x + b)
plt.show()
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfx = pd.read_csv("abxixas.csv")
dfy = pd.read_csv("ordenadas.csv")


tlsx = np.array(dfx['abxixas'])
tlsy = np.array(dfy['ordenadas'])


a , b = np.polyfit(tlsx, tlsy, 1)
plt.plot(tlsx, a*tlsx + b)
plt.ylim(-10 ,10 ,5)
plt.xlim(-5 , 5, 1)
plt.show()


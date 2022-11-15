#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import pickle 


ptsy = []
ptsx = []

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-5, 5])
ax.set_ylim([-10, 10])
plt.ylim(-10 ,10 ,5)
plt.xlim(-5 , 5, 1)



def onclick(event):
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       (event.button, event.x, event.y, event.xdata, event.ydata)) 
    plt.plot(event.xdata, event.ydata, '.')
    fig.canvas.draw()
    global ptsy
    ptsy.append([event.ydata])
    global ptsx
    ptsx.append([event.xdata])
    

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


dfx = pd.DataFrame(ptsx).reset_index(drop=True)
dfx = pd.DataFrame(ptsx,columns = ['abxixas'])
dfx.to_csv("abxixas.csv")


dfy = pd.DataFrame(ptsy).reset_index(drop=True)
dfy = pd.DataFrame(ptsy,columns = ['ordenadas'])
dfy.to_csv("ordenadas.csv")






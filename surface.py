#!/usr/bin/env python
# coding: utf-8

# In[22]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import pandas as pd



x= pd.read_csv('grid.csv')

drs=[]
drs.append(0.4)
for i in x['0.4']:
    drs.append(float(i))



lrs=[]
lrs.append(0.0001)
for i in x['0.0001']:
    lrs.append(float(i))

acs=[]
acs.append(0.9387)
for i in x['0.9387']:
    acs.append(float(i))

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Dropout Probability')
ax.set_zlabel('Accuracy')

ax.set_zticks([0.90, 0.91,0.92,0.93,0.94])
ax.set_xticks([0.0001,0.0003,0.0005,0.0007,0.0009,0.0011])
ax.set_yticks([0.40,0.45,0.50,0.55, 0.60])
surf = ax.plot_trisurf(lrs, drs, acs, cmap=cm.jet, linewidth=0.1,vmin=0.90, vmax=0.9325)

fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.savefig('teste.pdf')
plt.show()






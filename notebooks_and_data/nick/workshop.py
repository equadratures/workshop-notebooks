import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc
import os

#%%
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = 6,4
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
os.chdir('/home/nick/Documents/eq_workshop/Figures')

#%%
np.random.seed(0)
x = np.random.uniform(size=30)
y = 0.5 * x + 0.05 * np.random.randn(30)
plt.scatter(x,y)
plt.grid()
# plt.show()

plt.savefig('lin_regress0.eps')

#%%
np.random.seed(0)
x = np.random.uniform(size=30)
y = 0.5 * x + 0.05 * np.random.randn(30)
y_true = 0.5 * x
plt.scatter(x,y)
plt.plot(x,y_true,color='red')
plt.grid()
# plt.show()
plt.ylabel('y')
plt.xlabel('s')
plt.savefig('lin_regress1.eps')

#%%
np.random.seed(0)
x = np.random.uniform(size=30)
y = 0.5 * x + 0.05 * np.random.randn(30)
y_true = 0.5 * x
fig,ax = plt.subplots()
ax.scatter(x,y)
ax.plot(x,y_true,color='red')
lines = []
for i in range(30):
    lines.append([(x[i],y[i]), (x[i],y_true[i])])
lc = mc.LineCollection(lines, linewidths=1, color='black')
ax.add_collection(lc)
plt.grid()
# plt.show()
plt.ylabel('y')
plt.xlabel('s')
# plt.savefig('lin_regress2.eps')
#%%
np.random.seed(1)
x = np.random.uniform(size=30)
y = -0.5 * x + 0.5+  0.05 * np.random.randn(30)
y_true = -0.5 * x + 0.5
plt.scatter(x,y)
plt.plot(x,y_true,color='red')
plt.grid()
plt.show()
plt.ylabel('y')
plt.xlabel('s')
plt.savefig('lin_regress4.eps')

#%%
np.random.seed(1)
x = np.sort(np.random.uniform(size=30))
y = 2.0*x**2 -0.5 * x + 0.5+  0.15 * np.random.randn(30)
y_true = 2.0*x**2 -0.5 * x + 0.5
plt.scatter(x,y)
plt.grid()
plt.show()
plt.ylabel('y')
plt.xlabel('s')
plt.savefig('quad_regress0.eps')
#%%
np.random.seed(1)
x = np.sort(np.random.uniform(size=30))
y = 2.0*x**2 -0.5 * x + 0.5+  0.15 * np.random.randn(30)
y_true = 2.0*x**2 -0.5 * x + 0.5
plt.scatter(x,y)
plt.plot(x,y_true,color='red')
plt.grid()
plt.show()
plt.ylabel('y')
plt.xlabel('s')
plt.savefig('quad_regress.eps')